----
Waze Churn Prediction Project
---
----


Objective
---

The goal of this project is to predict which users are likely to churn (i.e., stop using the Waze app) based on their behavior and usage patterns. The key business objective is to maximize recall for churned users to allow for early intervention.

📈 Dataset Overview

Initial Users: 14,999

Final Users (after cleaning): 14,299

Initial Features: 13 columns

Final Features after feature engineering: 20 columns

Target Variable: label (Retained / Churned)

Churn Rate: 17.7%

🧹 Data Cleaning & Feature Engineering

Removed 700 users with missing target values.

Capped outliers using IQR in key columns like sessions, drives, duration_minutes_drives, etc.

Created 7 new features:

km_per_drive

km_per_driving_day

drives_per_driving_day

total_sessions_per_day

percent_sessions_in_last_month

percent_of_drives_to_favorite

professional_driver

📆 Exploratory Data Analysis (EDA)

Key Insights:

Retained users: 82.26% – Churned users: 17.74%

Churned users have slightly more sessions on average, but lower consistency.

Median sessions: Churned (59), Retained (56)

✍️ Recommended Visuals for Power BI:

Pie chart: Churn vs Retained distribution

Boxplots: Sessions, Drives, Total Sessions, Driven Distance by Label

Heatmap: Correlation matrix for numerical features

📚 Dataset Summary (Before Modeling)

Total users: 14,299

Columns: 20 (after feature engineering)

Summary Statistics (Recommended for Power BI KPIs):

Metric

Value

Total Users

14,299

Retention Rate

82.26%

Churn Rate

17.74%

Avg. Sessions

80.63

Avg. Drives

44.83

Avg. Distance Driven

303.67 km

Avg. Drive Duration

117.91 min

Avg. Activity Days

15.54

KPIs to be shown on Page 1 of Power BI Dashboard along with project summary text box.

🤖 Modeling

✅ Random Forest

Best Recall (Validation): 0.13

Accuracy: 0.82

Precision: 0.47

Not selected due to poor recall

📉 XGBoost (Final Model)

Best Recall (Validation): 0.14

Accuracy: 0.80

Precision: 0.36

⚠️ After Threshold Tuning (Threshold = 0.158):

Recall improved to: 0.50

Accuracy: 0.75

Precision: 0.35

F1 Score: 0.41

📄 Final Test Set Results (XGBoost with threshold tuning)

Metric

Value

Recall

0.50

Precision

0.35

F1 Score

0.41

Accuracy

0.75

Confusion Matrix



Predicted Churned

Predicted Retained

Actual Churned

254

253

Actual Retained

471

1882

Visual Confusion Matrix plot is recommended on Page 2 of Power BI Dashboard.

📊 Feature Importance (Top Predictors)

Drives

Total Sessions

Activity Days

KM per Drive

Include xgb.plot_importance() bar chart on Page 2 of Power BI Dashboard.

⚖️ Threshold Tuning Details

Implemented custom function to evaluate recall at multiple thresholds.

Selected 0.158 as the optimal threshold for maximizing recall.

Post-tuning classification report:

Churned Precision: 0.35

Churned Recall: 0.50

Churned F1: 0.41

🔐 Next Steps

Integrate the model into a user retention alert system.

Track churn prediction accuracy over time.

Consider retraining monthly to capture seasonal patterns.

📦 Dashboard Layout (Power BI Proposal)

Page 1: Executive Summary + KPIs

Text card: Goal & summary

KPIs:

Total Users

Churn Rate

Retention Rate

Avg. Sessions / Drives / Activity Days / Distance

Page 2: Churn Behavior Analysis

Pie chart: Churn vs Retained

Box plots: sessions, drives, total_sessions, driven_km_drives by label

Confusion Matrix

Feature Importance bar chart

Page 3 (Optional): Time-based or Device-based Insights

Avg. sessions per device

Activity by onboarding day

🚀 Deployment

The model is ready to be deployed as part of a churn-monitoring solution. Consider integrating it via an API into a CRM system.

