

# Sure Tomorrow — ML + Data Obfuscation 🔐🤖

This project explores how **machine learning** can support an insurance company (*Sure Tomorrow*) across four tasks:  
customer similarity search, benefits classification, benefits regression, and **data obfuscation** to protect personal data while keeping models useful.

---

## What’s inside 🧩

1) **Similar customers (kNN / Nearest Neighbors)**  
   Find customers with similar profiles to support marketing and customer support.

2) **Classification: will the customer receive benefits? ✅/❌**  
   Predict whether a customer is likely to receive insurance benefits.

3) **Regression: how many benefits? 📈**  
   Predict the *number* of benefits a customer may receive.

4) **Data obfuscation (privacy) 🔒**  
   Transform the feature matrix with an **invertible matrix** so the data is harder to interpret, but can still be used for modeling.

---

## Dataset 📦

File: `insurance_us.csv`

Key features:
- `gender`
- `age`
- `income`
- `family_members`

Targets:
- `insurance_benefits_received` (classification: received benefits or not)
- `insurance_benefits` (regression: number of benefits)

---

## Results (from the notebook) 🧪

### Classification (kNN vs baseline)
- **kNN:** Accuracy **0.97**, AUC **1.00**, F1 **0.883**
- **Dummy baseline:** Accuracy **0.113**, AUC **0.50**, F1 **0.203**

➡️ The baseline behaves like random guessing (AUC ≈ 0.5), while kNN shows strong predictive signal.

### Regression (Linear Regression)
- **RMSE:** **0.333**
- **R²:** **0.413**

---

## Privacy check (obfuscation) 🔐

We generate a random **invertible** matrix `P` and transform features:

- `X' = X · P`  
- Recover original: `X = X' · P⁻¹`

The notebook verifies exact recovery:
- **Recovery exact:** `True`

---

## Tech stack 🛠️

- Python
- NumPy, pandas
- scikit-learn (NearestNeighbors, KNN, metrics, train/test split, DummyClassifier)

---

## How to run ▶️

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
