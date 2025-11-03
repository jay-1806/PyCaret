# PyCaret Assignment

This repository contains my Colab notebooks for the PyCaret Assignment.  
Each notebook demonstrates a different machine learning task using PyCaret’s low-code AutoML framework.  
All notebooks were executed in my own Colab environment with outputs saved in the `.ipynb` files.

---

## Contents

### 1. Binary Classification — Diabetes Prediction
**Notebook:** `binary_classification.ipynb`  
**Dataset:** `diabetes_dataset.csv`  
**Goal:** Predict whether a patient is diagnosed with diabetes.  

**Highlights:**
- Uses `pycaret.classification`
- Automatically handles encoding, normalization, and train/test split
- Steps: `setup` → `compare_models` → `tune_model` → `finalize_model` → `predict_model`
- Evaluates model performance with accuracy and AUC

---

### 2. Multiclass Classification — Student Performance
**Notebook:** `multiclass.ipynb`  
**Dataset:** `Student_performance_data.csv`  
**Goal:** Predict student grade classes across multiple categories.  

**Highlights:**
- Uses `pycaret.classification`
- Evaluates models using Accuracy and F1-score
- Visualizes confusion matrix
- Saves the best model for reuse

---

### 3. Regression — Car Price Prediction
**Notebook:** `regression.ipynb`  
**Dataset:** `CarPrice_Assignment.csv`  
**Goal:** Predict car prices based on vehicle specifications.  

**Highlights:**
- Uses `pycaret.regression`
- Normalizes features and applies feature selection
- Compares multiple regression algorithms
- Tunes and finalizes the best model

---

### 4. Clustering — Customer Segmentation
**Notebook:** `clustering.ipynb`  
**Dataset:** `segmentation_data.csv`  
**Goal:** Segment customers into meaningful groups based on behavior and attributes.  

**Highlights:**
- Uses `pycaret.clustering`
- Creates a K-Means clustering model
- Optionally tunes cluster count using silhouette score
- Assigns cluster labels and visualizes groups

---

### 5. Anomaly Detection — Bank Transactions
**Notebook:** `Anomaly_detection.ipynb`  
**Dataset:** `bank_transactions_data.csv`  
**Goal:** Detect unusual or suspicious bank transactions.  

**Highlights:**
- Uses `pycaret.anomaly` with Isolation Forest
- Generates anomaly scores and binary flags
- Supports contamination adjustment to control the anomaly rate
- Saves the anomaly detection model for reuse

---

### 6. Association Rule Mining — Groceries Transactions
**Notebook:** `association_rule_mining.ipynb`  
**Dataset:** `Groceries_dataset.csv`  
**Goal:** Find product combinations that are frequently bought together.  

**Highlights:**
- Uses the Apriori algorithm for rule mining
- Computes Support, Confidence, and Lift for each rule
- Visualizes the strongest rules
- Exports rules to CSV for analysis

---

### 7. Time Series Forecasting — Delhi Climate Data
**Notebook:** `time_series.ipynb`  
**Dataset:** `DailyDelhiClimateTrain.csv`  
**Goal:** Forecast future temperature values using past climate data.  

**Highlights:**
- Uses `pycaret.time_series`
- Detects frequency and seasonality automatically
- Compares models such as ARIMA, ETS, and Prophet
- Forecasts future temperature values (e.g., next 14 days)
- Includes visualization of predicted trends

---

### Notes
- All notebooks use PyCaret’s low-code approach for quick prototyping.  
- Each workflow includes automatic preprocessing, model comparison, tuning, and saving for deployment.
