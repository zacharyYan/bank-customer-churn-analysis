# Bank Customer Churn Prediction

An end-to-end supervised learning portfolio project that predicts customer churn and evaluates how classification thresholds change the trade-off between finding more at-risk customers and avoiding unnecessary outreach.

## Highlights

- Analyzed 10,000 anonymized customer records with a 20.37% churn rate.
- Built leakage-safe preprocessing for numeric and categorical features.
- Compared logistic regression, K-nearest neighbors, and random forest models.
- Tuned the random forest with five-fold cross-validation.
- Achieved a test ROC-AUC of **0.862** on an untouched 2,000-customer test set.
- At a 0.50 threshold, achieved **73.2% recall**, **53.2% precision**, and **0.616 F1**.
- Used held-out permutation importance to identify features associated with churn risk.

## Business Question

How can a bank prioritize limited retention resources toward customers who are more likely to leave while making the precision-recall trade-off explicit?

## Methods

Python, pandas, scikit-learn, stratified train/validation/test splitting, one-hot encoding, feature scaling, grid search, ROC-AUC, precision, recall, F1, threshold analysis, and permutation importance.

## Repository Contents

- `Bank_Customer_Churn_Portfolio.ipynb` — executed analysis with explanations and results
- `requirements.txt` — Python dependencies

## Data and Privacy

The course-provided dataset is not redistributed in this repository because its redistribution license was not confirmed. It also contains direct identifier-like fields (`RowNumber`, `CustomerId`, and `Surname`) that are excluded from modeling.

To reproduce the notebook, place your authorized copy of the 10,000-row dataset in the repository root as `bank.data.csv`. The expected target column is `Exited`.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab Bank_Customer_Churn_Portfolio.ipynb
```

## Limitations

- This is a portfolio analysis, not a deployed banking system.
- Model importance describes predictive association, not causation.
- A real retention threshold should incorporate customer value, intervention cost, and outreach capacity.
