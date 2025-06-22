---
Waze Churn Prediction Project
---
-----

Predicting user churn based on app activity and driving behavior.

---

Objective
----

The goal of this project is to predict which users are likely to churn (i.e., stop using the Waze app) based on their activity patterns, and maximize **recall** for churned users, to assist the business in **early intervention** strategies.

---

Dataset Overview
---

- **Size:** 14,999 users
- **Features:** 13 initial columns, later expanded to 20 after feature engineering
- **Target variable:** `label` (Retained / Churned)
- **Missing values:** 700 missing in the target column were removed

---

Data Cleaning & Preprocessing
---

- Removed missing values from target column
- Capped outliers in key columns using the IQR method
- Created 7 new features (e.g., `km_per_drive`, `km_per_driving_day`, `professional_driver`)
- Final dataset shape: **(14,299, 20)**

---

Exploratory Data Analysis (EDA)
---

- **Churn Rate:** 17.7%
- Churned users tend to drive slightly more but less frequently
- Median comparison by label and correlation heatmap were analyzed

> *Recommendation:* Add **EDA plots** like:
> - Pie chart for churn distribution  
> - Boxplots comparing sessions/drives per label  
> - Correlation heatmap  

---

Modeling & Evaluation
---

Two models were trained and compared:

### **Random Forest**
- Best Recall (validation): **0.13**
- Accuracy: **0.81**
- Precision: 0.46  
  *Not chosen as final model due to low recall*

### 2. **XGBoost**
- Best Recall (validation): **0.14**
- Accuracy: **0.80**
- Precision: 0.36

**After threshold tuning (`threshold = 0.158`)**
- **Recall improved to 0.50**
- **Accuracy:** 0.75
- **Precision:** 0.35
- **F1-score:** 0.41

---

Final Model: XGBoost + Threshold Tuning
---

### Final Results on Test Set:

| Metric    | Value   |
|-----------|---------|
| Recall    | **0.50** |
| Precision | 0.35    |
| F1        | 0.41    |
| Accuracy  | 0.75    |

*This threshold allowed for **better identification of churned users** with reasonable balance of precision.*

---

Confusion Matrix
---

> *Recommendation:* Include the **Confusion Matrix plot** for the final model  
> Example:
> - TP, TN, FP, FN counts
> - Visual comparison of misclassifications

---

Feature Importance
---

- The most important features identified by XGBoost were:
  - `drives`
  - `total_sessions`
  - `activity_days`
  - `km_per_drive`

Include bar chart from `plot_importance(xgb)` here

---

Threshold Tuning
---

- A custom threshold finder function was implemented to **maximize recall**
- Best threshold: **0.158**
- Actual Recall: **0.501**

---
