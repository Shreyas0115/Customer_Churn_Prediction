# 📉 Customer Churn Prediction 

## 📂 Input
- A **CSV file** with customer information.  
- Must contain a churn column (`churn`, `Churn`, `Exited`, `exit`, or `is_churn`) — auto-detected if not specified.  

---

## 🚀 Steps Performed in the Code

### 1. Load & Identify Target Column
- Read the dataset using **pandas**.  
- Automatically infer the churn column if not explicitly given.  
- Target (`y`) is cast to integer (0/1).  
- Remaining columns are split into:  
  - **Numeric features** (numbers like age, balance, tenure)  
  - **Categorical features** (strings like geography, gender, subscription plan)  

**What I learned**: Auto-detecting the target makes the script flexible. Splitting features ensures correct preprocessing for each type.

---

### 2. Preprocessing with ColumnTransformer
- **Numeric pipeline**:  
  - Fill missing values with median (`SimpleImputer`).  
  - Scale values with `StandardScaler`.  
- **Categorical pipeline**:  
  - Fill missing values with most frequent value.  
  - Encode categories into one-hot vectors (`OneHotEncoder`).  
- Combine both using `ColumnTransformer`.

**What I learned**: Handling numeric and categorical columns differently is crucial. Pipelines avoid manual preprocessing errors.

---

### 3. Train Models
- Three classifiers are trained (each inside a pipeline with preprocessing):  
  - **Logistic Regression** (with class balancing)  
  - **Random Forest** (400 trees, balanced subsample)  
  - **Gradient Boosting Classifier**  
- Each model is trained on the training set.

**What I learned**: Training multiple models in parallel helps find the best performer. Balanced class weights address churn imbalance.

---

### 4. Evaluate Models
- Metric: **ROC-AUC** (captures ranking ability even with imbalanced data).  
- Best model is selected by highest ROC-AUC.  
- Classification report and confusion matrix are printed for the chosen model.  

**What I learned**: ROC-AUC is more reliable than accuracy when churn cases are rare. Confusion matrix shows where the model struggles.

---

### 5. Interpretability (Optional)
- For models that support feature importance, perform **Permutation Importance** on test data.  
- Map feature importances back to original column names (numerical + one-hot encoded categorical).  
- Print top 20 important features.  

**What I learned**: Interpretability is key in churn prediction — knowing **why** a customer is likely to churn is as valuable as the prediction.

---

### 6. Save Best Model
- Save the best pipeline and metadata using **joblib**.  
- Metadata includes:  
  - Best model type  
  - Target column name  
  - List of numeric and categorical features  
  - Test split size  

**What I learned**: Saving metadata makes the model reusable and transparent. Pipelines ensure preprocessing is included inside the saved model.

---
