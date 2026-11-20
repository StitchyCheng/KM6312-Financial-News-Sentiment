"""KM6312 methodology: EDA, six experiments, and paired trading-day bootstrap.
Run: python KM6312_methodology_final.py --input 'company day features.xlsx'
Dependencies: pandas numpy openpyxl matplotlib scikit-learn
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, RocCurveDisplay

SENTIMENT=['Positive_Ratio','Negative_Ratio','News_Count','HasNews']
MARKET=['Return','Volume_Change','Volatility_5D']
FEATURES={'Sentiment-only':SENTIMENT,'Market-only':MARKET,'Combined':SENTIMENT+MARKET}
RANDOM_SEED=42

def score(y, p, pred):
    return {'ROC_AUC':roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
            'Macro_F1':f1_score(y,pred,average='macro',zero_division=0),
            'Accuracy':accuracy_score(y,pred),
            'Precision':precision_score(y,pred,zero_division=0),
            'Recall':recall_score(y,pred,zero_division=0)}

def bootstrap_day(y, p_comb, p_market, dates, n_boot=1000, seed=42):
    """Resample distinct days with replacement, retaining all companies per sampled day."""
    rng=np.random.default_rng(seed)
    days, code=np.unique(np.asarray(dates),return_inverse=True)
    order=np.argsort(code)
    counts=np.bincount(code)
    starts=np.r_[0,np.cumsum(counts)[:-1]]
    diffs=[]
    for _ in range(n_boot):
        sampled=rng.integers(0,len(days),size=len(days))
        idx=np.concatenate([order[starts[k]:starts[k]+counts[k]] for k in sampled])
        if np.unique(y[idx]).size<2: continue
        diffs.append(roc_auc_score(y[idx],p_comb[idx])-roc_auc_score(y[idx],p_market[idx]))
    diffs=np.asarray(diffs)
    return {'ROC_AUC_Delta':roc_auc_score(y,p_comb)-roc_auc_score(y,p_market),
            'CI_Lower_95':np.quantile(diffs,.025),'CI_Upper_95':np.quantile(diffs,.975),
            'Bootstrap_Replicates':len(diffs)}

def main(input_file, output_dir, bootstrap_iters):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_excel(input_file)
    required={'Ticker','Trading_Date','Split','Label','Next_Return','Average_Sentiment_Score','Included'} | set(SENTIMENT+MARKET)
    missing=required-set(df.columns)
    if missing:raise ValueError(f'Missing columns: {sorted(missing)}')
    df['Trading_Date']=pd.to_datetime(df.Trading_Date)
    df=df.loc[df.Included.eq(True)].copy()
    df=df.loc[df.Split.isin(['Training','Validation','Test'])].copy()
    df=df.sort_values(['Trading_Date','Ticker']).reset_index(drop=True)
    if df.duplicated(['Ticker','Trading_Date']).any():raise ValueError('Duplicate ticker-date rows')
    if not np.isfinite(df[SENTIMENT+MARKET].to_numpy(dtype=float)).all():raise ValueError('Non-finite features')
    expected={'Training':(2018,2021),'Validation':(2022,2022),'Test':(2023,2023)}
    for split,(lo,hi) in expected.items():
        years=df.loc[df.Split.eq(split),'Trading_Date'].dt.year
        if years.empty or not years.between(lo,hi).all():raise ValueError(f'Unexpected dates in {split}')
    if 'Cross_Split_Boundary' in df and df.Cross_Split_Boundary.any():raise ValueError('Cross-split target leakage')
    if 'Next_Trading_Date' in df:
        nxt=pd.to_datetime(df.Next_Trading_Date)
        if (nxt<=df.Trading_Date).any():raise ValueError('Target date must be later')
    if not np.array_equal(df.Label.to_numpy(),(df.Next_Return.to_numpy()>0).astype(int)):
        raise ValueError('Target label mismatches next return')
    stats=df.groupby('Split').agg(Company_Days=('Label','size'),Companies=('Ticker','nunique'),News=('News_Count','sum'),Up=('Label','sum')).reindex(['Training','Validation','Test'])
    stats['Up_Rate']=stats.Up/stats.Company_Days
    stats.to_csv(out/'dataset_summary.csv')
    print('Dataset summary:\n',stats)

    # 3.1 EDA
    distribution=df.groupby(['Split','Label']).size().unstack(fill_value=0)
    distribution.columns=['Down_or_Flat','Up'];distribution.to_csv(out/'eda_direction_counts.csv')
    fig,ax=plt.subplots(figsize=(7,4));distribution.plot.bar(stacked=True,ax=ax);ax.set_title('Next-day stock direction by split');ax.set_xlabel('');ax.set_ylabel('Company-days');fig.tight_layout();fig.savefig(out/'eda_direction.png',dpi=180);plt.close(fig)
    news=df.News_Count
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(news.clip(upper=news.quantile(.99)),bins=35);ax.set_title('News count (clipped at 99th percentile for display)');ax.set_xlabel('News per company-day');ax.set_ylabel('Company-days');fig.tight_layout();fig.savefig(out/'eda_news_count.png',dpi=180);plt.close(fig)
    df['Sentiment_Group']=np.select([df.HasNews.eq(0),df.Average_Sentiment_Score.gt(0),df.Average_Sentiment_Score.lt(0)],['No news','Positive','Negative'],default='Neutral')
    groups=df.groupby(['Split','Sentiment_Group']).agg(Company_Days=('Label','size'),Mean_Next_Return=('Next_Return','mean'),Up_Rate=('Label','mean')).reset_index()
    groups.to_csv(out/'eda_sentiment_vs_next_day.csv',index=False)
    testgroups=groups[groups.Split.eq('Test')].set_index('Sentiment_Group').reindex(['No news','Negative','Neutral','Positive']).dropna(subset=['Up_Rate'])
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(testgroups.index,testgroups.Up_Rate);ax.set_ylim(0,1);ax.set_title('Next-day up rate by news sentiment (2023 test)');ax.set_ylabel('Up rate');fig.tight_layout();fig.savefig(out/'eda_sentiment_up_rate_test.png',dpi=180);plt.close(fig)

    # 3.2-3.3: select hyperparameters on validation only (never on test).
    fit=df[df.Split.eq('Training')]; valid=df[df.Split.eq('Validation')]; test=df[df.Split.eq('Test')]
    y_train=fit.Label.to_numpy(); y_val=valid.Label.to_numpy(); y_test=test.Label.to_numpy()
    model_candidates={
        'Logistic Regression': [
            ('C=0.1', lambda:Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',LogisticRegression(C=.1,max_iter=1500,random_state=RANDOM_SEED))])),
            ('C=1', lambda:Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',LogisticRegression(C=1,max_iter=1500,random_state=RANDOM_SEED))]))],
        'Random Forest': [
            ('leaf=20',lambda:RandomForestClassifier(n_estimators=160,min_samples_leaf=20,max_features='sqrt',n_jobs=-1,random_state=RANDOM_SEED)),
            ('leaf=50',lambda:RandomForestClassifier(n_estimators=160,min_samples_leaf=50,max_features='sqrt',n_jobs=-1,random_state=RANDOM_SEED))]}
    results=[]; predictions=test[['Ticker','Trading_Date','Label','Next_Return']].copy()
    params=[]; all_probs={}
    for method,candidates in model_candidates.items():
        for group,cols in FEATURES.items():
            best=None
            for param,maker in candidates:
                model=maker(); model.fit(fit[cols],y_train)
                valid_prob=model.predict_proba(valid[cols])[:,1]
                valauc=roc_auc_score(y_val,valid_prob)
                if best is None or valauc>best[0]:best=(valauc,param,model)
            valauc,param,model=best
            key=f'{method} | {group}'
            prob=model.predict_proba(test[cols])[:,1]
            pred=(prob>=.5).astype(int)
            row={'Algorithm':method,'Feature_Set':group,'Validation_ROC_AUC':valauc,'Selected_Parameter':param,**score(y_test,prob,pred)}
            results.append(row);params.append({'Algorithm':method,'Feature_Set':group,'Parameter':param})
            safe=f'{method.replace(" ","_")}_{group}'
            predictions[f'{safe}_Probability']=prob
            predictions[f'{safe}_Prediction']=pred
            all_probs[(method,group)]=prob
            cm=confusion_matrix(y_test,pred,labels=[0,1]);pd.DataFrame(cm,index=['Actual_0','Actual_1'],columns=['Predicted_0','Predicted_1']).to_csv(out/f'confusion_{safe}.csv')
            fig,ax=plt.subplots(figsize=(4,4));ax.imshow(cm,cmap='Blues');ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(key)
            for i in range(2):
                for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
            fig.tight_layout();fig.savefig(out/f'confusion_{safe}.png',dpi=160);plt.close(fig)
            print(key, 'AUC',round(row['ROC_AUC'],4),'val AUC',round(valauc,4))
    res=pd.DataFrame(results);res.to_csv(out/'six_model_results.csv',index=False)
    pd.DataFrame(params).to_csv(out/'selected_hyperparameters.csv',index=False)
    predictions.to_csv(out/'test_predictions.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4));names=[f'{r.Algorithm[:2]} / {r.Feature_Set}' for r in res.itertuples()];ax.bar(names,res.ROC_AUC);ax.axhline(.5,ls='--',c='gray',label='Chance ROC-AUC');ax.set_ylabel('Test ROC-AUC');ax.set_ylim(0,1);ax.tick_params(axis='x',rotation=30);ax.legend();fig.tight_layout();fig.savefig(out/'six_model_auc.png',dpi=180);plt.close(fig)

    # 3.4 Pair day bootstrap: test data only, same resampled days for both models.
    comparisons=[]
    for method in model_candidates:
        b=bootstrap_day(y_test,all_probs[(method,'Combined')],all_probs[(method,'Market-only')],test.Trading_Date.to_numpy(),bootstrap_iters,RANDOM_SEED)
        comparisons.append({'Algorithm':method,**b})
    pd.DataFrame(comparisons).to_csv(out/'paired_day_bootstrap.csv',index=False)
    # baselines for contextualizing RQ1
    majority=int(fit.Label.mean()>=.5)
    baseline=score(y_test,np.repeat(fit.Label.mean(),len(y_test)),np.full(len(y_test),majority,dtype=int))
    pd.DataFrame([{'Baseline':'Training-prior/majority',**baseline}]).to_csv(out/'baseline_results.csv',index=False)
    print('Bootstrap comparison:\n',pd.DataFrame(comparisons).to_string(index=False))
    print('Outputs:',out.resolve())

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='company day features.xlsx')
    parser.add_argument('--out',default='KM6312_methodology_results')
    parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args()
    main(args.input,args.out,args.bootstrap)
