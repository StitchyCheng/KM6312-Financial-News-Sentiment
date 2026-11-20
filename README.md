# Financial News Sentiment and Stock Direction Prediction

## 1. Project Overview

This project investigates whether financial news sentiment can predict next-day stock price direction and whether sentiment features provide incremental predictive value beyond historical market information.

## 2. Dataset

- Source: FNSPID
- Companies: 50 S&P 500 companies
- Study period: 2018–2023
- Unit of analysis: Company-day

## 3. Methodology

Two classification algorithms were employed:

- Logistic Regression
- Random Forest

Three feature sets were evaluated:

- Sentiment-only
- Market-only
- Combined

The dataset was divided chronologically into training (2018–2021), validation (2022), and testing (2023) periods.

## 4. Model Evaluation

- ROC-AUC
- Macro-F1
- Accuracy, Precision and Recall
- Paired Bootstrap with 1,000 resamples and 95% confidence intervals

## 5. Repository Structure

- `data/`: Processed datasets
- `code/`: Data processing and modelling scripts
- `results/`: Experimental results and figures

## 6. How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the main modelling script:

```bash
python code/KM6312_methodology_final.py
```

## 7. References

Araci (2019), Breiman (2001), Pedregosa et al. (2011), and the FNSPID dataset paper.

## 8. Course

KM6312 Information Mining & Analysis
