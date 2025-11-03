# PyCaret Assignment

This repository contains my Colab notebooks for the **PyCaret Assignment**.  
Each notebook demonstrates a different machine learning task using PyCaret’s low-code AutoML framework.  
All notebooks were executed in my own Colab environment with outputs saved in the `.ipynb` files.

---

## Contents

### 1. Binary Classification — Loan Default Prediction
**Notebook:** `pycaret_binary_classification.ipynb`  
**Dataset:** [Loan Default Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/zaurbegiev/my-dataset)  
**Goal:** Predict whether a customer will default on a loan (target = `default`)  
**Highlights:**  
- Uses `pycaret.classification`  
- GPU enabled with `use_gpu=True`  
- Steps: setup → compare_models → tune_model → finalize_model → predict_model  

---

### 2. Multiclass Classification — Dry Bean Dataset
**Notebook:** `pycaret_multiclass_classification.ipynb`  
**Dataset:** [Dry Bean Dataset (UCI via Kaggle)](https://www.kaggle.com/datasets/uciml/dry-bean-dataset)  
**Goal:** Predict the bean type (7 possible classes) based on physical properties.  
**Highlights:**  
- Uses `pycaret.classification`  
- Evaluates F1-macro, accuracy, and confusion matrix  
- Shows model comparison and best model saving  

---

### 3. Regression — Student Performance Dataset
**Notebook:** `pycaret_regression.ipynb`  
**Dataset:** [Student Performance Dataset (Kaggle)](https://www.kaggle.com/datasets/whenamancodes/student-performance)  
**Goal:** Predict students’ final grade (`G3`) based on study habits and background features.  
**Highlights:**  
- Uses `pycaret.regression`  
- Demonstrates setup with `use_gpu=True` and feature normalization  
- Includes model tuning and prediction on test data  

---

### 4. Clustering — Mall Customers Segmentation
**Notebook:** `pycaret_clustering.ipynb`  
**Dataset:** [Mall Customer Segmentation Data (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)  
**Goal:** Group customers into distinct clusters based on spending score and annual income.  
**Highlights:**  
- Uses `pycaret.clustering`  
- Visualizes cluster distribution  
- Displays PCA and t-SNE plots for interpretation  

---

### 5. Anomaly Detection — Credit Card Fraud
**Notebook:** `pycaret_anomaly_detection.ipynb`  
**Dataset:** [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Goal:** Detect fraudulent transactions from anonymized card data.  
**Highlights:**  
- Uses `pycaret.anomaly`  
- GPU enabled; compares multiple unsupervised models  
- Marks anomalous transactions and visualizes their distribution  

---

### 6. Association Rule Mining — Online Retail Transactions
**Notebook:** `pycaret_association_rules.ipynb`  
**Dataset:** [Online Retail II Dataset (Kaggle)](https://www.kaggle.com/datasets/carrie1/ecommerce-data)  
**Goal:** Discover item co-purchase patterns using association rules (support, confidence, lift).  
**Highlights:**  
- Uses PyCaret 2.3.5 (`!pip install pycaret==2.3.5`)  
- Applies Apriori algorithm and visualizes top rules  
- Focuses on interpretability of rules and lift ratio  

---

### 7. Time Series Forecasting (Univariate) — Daily Minimum Temperatures
**Notebook:** `pycaret_timeseries_univariate.ipynb`  
**Dataset:** [Daily Minimum Temperatures Dataset (Kaggle)](https://www.kaggle.com/datasets/abhinavralhan/daily-minimum-temperatures-in-melbourne)  
**Goal:** Forecast future temperatures based on a single time series.  
**Highlights:**  
- Uses `pycaret.time_series`  
- Auto-detects frequency and seasonality  
- Compares ARIMA, ETS, Prophet, and AutoETS models  

---

### 8. Time Series Forecasting (Univariate + Exogenous) — Energy Consumption
**Notebook:** `pycaret_timeseries_univariate_exogenous.ipynb`  
**Dataset:** [Household Electric Power Consumption (Kaggle)](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)  
**Goal:** Forecast daily power usage using weather features as exogenous variables.  
**Highlights:**  
- Uses `pycaret.time_series`  
- Adds temperature and humidity as regressors  
- Evaluates models on RMSE and MAPE metrics  

---

## Folder Structure

pycaret-assignment/
├─ README.md
├─ pycaret_binary_classification.ipynb
├─ pycaret_multiclass_classification.ipynb
├─ pycaret_regression.ipynb
├─ pycaret_clustering.ipynb
├─ pycaret_anomaly_detection.ipynb
├─ pycaret_association_rules.ipynb
├─ pycaret_timeseries_univariate.ipynb
└─ pycaret_timeseries_univariate_exogenous.ipynb


---

## How to Run

1. Open any notebook in Google Colab.  
2. Install PyCaret (use version 2.3.5 only for the association rules notebook).  
   ```python
   !pip install pycaret
   # or
   !pip install pycaret==2.3.5


Evaluation Notes

Each notebook shows complete model comparison outputs.

Datasets are different from PyCaret’s example datasets, per assignment requirements.

Each notebook is self-contained and re-runnable.

Videos explaining each notebook are stored separately in another folder.

References

PyCaret Official Documentation: https://pycaret.gitbook.io/docs

PyCaret Tutorials: https://pycaret.gitbook.io/docs/get-started/tutorials

Kaggle Datasets used above

PyCaret Examples Repository: https://github.com/pycaret/examples
