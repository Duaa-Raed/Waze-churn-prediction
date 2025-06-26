# ==================================================================================
# # 1. Import Libraries and Tools
# ==================================================================================

# Import data handling libraries
import numpy as np
import pandas as pd

# Import data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import modeling libraries
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

# Import statistical analysis libraries
from scipy import stats

# ==================================================================================
# # 2. Load and Initially Explore Data
# ==================================================================================

# Load the dataset
# Note: Please change the path to the correct path on your machine
try:
    waze = pd.read_csv("C:/Users/duaar/OneDrive/Desktop/Waze/waze_dataset.csv")
except FileNotFoundError:
    print("Error: File not found. Please check the specified path.")
    # You can place an alternative path here or terminate the program
    # exit()

# Display a sample of the data
print("### Data Sample (first 10 rows):")
print(waze.head(10))

# Display information about columns and data types
print("\n### Data Information (Info):")
waze.info()

# Display descriptive statistics for numerical columns
print("\n### Descriptive Statistics for Data:")
print(waze.describe())

# Check for missing values
print("\n### Number of missing values in each column:")
print(waze.isnull().sum())


# ==================================================================================
# # 3. Data Cleaning and Feature Engineering
# ==================================================================================

# Create a copy of the data to avoid modifying the original
df = waze.copy()

# --- 3.1. Handling Missing Values ---
# Drop rows with missing values
df.dropna(inplace=True)
print(f"\n### Data size after dropping missing values: {df.shape}")

# --- 3.2. Feature Engineering ---
print("\n### Creating new features...")

# Calculate driving-related rates
df['km_per_drive'] = df['driven_km_drives'] / df['drives']
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']
df['drives_per_driving_day'] = df['drives'] / df['driving_days']

# Calculate percentage of sessions in the last month
df['percent_sessions_in_last_month'] = df['sessions'] / df['total_sessions']

# Identify professional drivers
df['professional_driver'] = np.where((df['drives'] >= 60) & (df['driving_days'] >= 15), 1, 0)

# Calculate daily session rate
df['total_sessions_per_day'] = df['total_sessions'] / df['n_days_after_onboarding']

# Calculate speed (km/h)
df['km_per_hour'] = df['driven_km_drives'] / (df['duration_minutes_drives'] / 60)

# Calculate percentage of drives to favorite locations
df['percent_of_drives_to_favorite'] = (df['total_navigations_fav1'] + df['total_navigations_fav2']) / df['total_sessions']

# --- 3.3. Data Cleaning After Feature Engineering ---
# Handle infinite values (inf) that may result from division by zero
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Fill any remaining NaN values with zero (another strategy like mean or median can be chosen)
df.fillna(0, inplace=True)

# --- 3.4. Handling Outliers ---
# Identify columns to check
outlier_cols = [
    'sessions', 'drives', 'total_sessions', 'driven_km_drives',
    'duration_minutes_drives', 'total_navigations_fav1', 'total_navigations_fav2',
    'km_per_drive', 'km_per_driving_day', 'drives_per_driving_day'
]

# Handle outliers by capping at the 95th percentile
print("\n### Handling outliers...")
for col in outlier_cols:
    threshold = df[col].quantile(0.95)
    df.loc[df[col] > threshold, col] = threshold
    print(f"Capped outliers in column '{col}' at value {threshold:.2f}")

# --- 3.5. Encoding Categorical Variables ---
# Convert 'label' variable to numeric (0 for retained, 1 for churned)
df['label2'] = np.where(df['label'] == 'retained', 0, 1)

# Convert 'device' variable to numeric (0 for Android, 1 for iPhone)
df['device2'] = np.where(df['device'] == 'Android', 0, 1)

# --- 3.6. Data Type Conversion and Removing Unnecessary Columns ---
# Remove original non-numeric and unnecessary columns for modeling
df_model = df.drop(['ID', 'label', 'device'], axis=1)

print("\n### Information about the final data ready for modeling:")
df_model.info()


# ==================================================================================
# # 4. Exploratory Data Analysis (EDA)
# ==================================================================================
print("\n\n# ==================================================================================")
print("# 4. Exploratory Data Analysis (EDA)")
print("# ==================================================================================\n")

# --- 4.1. Target Variable Distribution ---
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=df)
plt.title('User Status Distribution (Churned vs. Retained)')
plt.savefig(r'C:/Users/duaar/OneDrive/Desktop/image/User_Status_Distribution_(Churned_vs._Retained).png')
plt.show()
print("### Status Distribution Percentage:")
print(df['label'].value_counts(normalize=True) * 100)

# --- 4.2. Correlation Matrix ---
plt.figure(figsize=(15, 12))
corr = df_model.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.savefig(r'C:/Users/duaar/OneDrive/Desktop/image/Feature_Correlation_Matrix.png')
plt.show()

# --- 4.3. Median Analysis by User Status ---
print("\n### Median Values for Different Features by User Status:")
numeric_cols = df.select_dtypes(include=np.number).columns.drop('label2')
print(df.groupby('label')[numeric_cols].median())


# ==================================================================================
# # 5. Data Preparation for Modeling
# ==================================================================================
print("\n\n# ==================================================================================")
print("# 5. Data Preparation for Modeling")
print("# ==================================================================================\n")

# --- 5.1. Define Features (X) and Target Variable (y) ---
X = df_model.drop('label2', axis=1)
y = df_model['label2']

# --- 5.2. Split Data into Training, Validation, and Test Sets (60-20-20) ---
# Initial split into 60% training and 40% temporary
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# Split the temporary set into 50% validation and 50% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# --- 5.3. Data Scaling ---
# Used primarily with models sensitive to feature scales like Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ==================================================================================
# # 6. Model Building and Training
# ==================================================================================
print("\n\n# ==================================================================================")
print("# 6. Model Building and Training")
print("# ==================================================================================\n")

# --- 6.1. Define Helper Functions for Evaluation ---
def make_results(model_name:str, model_object, metric:str):
    '''
    Extracts cross-validation results from a GridSearchCV object.
    '''
    metric_dict = {'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'}
    cv_results = pd.DataFrame(model_object.cv_results_)
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
    table = pd.DataFrame({'model': [model_name], 'precision': [precision],
                          'recall': [recall], 'F1': [f1], 'accuracy': [accuracy]})
    return table

def get_test_scores(model_name:str, preds, y_true):
    '''
    Calculates performance metrics on a dataset (validation or test).
    '''
    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    table = pd.DataFrame({'model': [model_name], 'precision': [precision],
                          'recall': [recall], 'F1': [f1], 'accuracy': [accuracy]})
    return table

# --- 6.2. Random Forest Model ---
print("\n--- Training Random Forest Model ---")
rf = RandomForestClassifier(random_state=42)
cv_params_rf = {'max_depth': [None], 'max_features': [1.0], 'max_samples': [1.0],
                'min_samples_leaf': [2], 'min_samples_split': [2], 'n_estimators': [300]}
scoring = ['accuracy', 'precision', 'recall', 'f1']
rf_cv = GridSearchCV(rf, cv_params_rf, scoring=scoring, cv=4, refit='recall')
rf_cv.fit(X_train, y_train)

print("Best parameters for Random Forest:", rf_cv.best_params_)

# Evaluate on validation set
rf_val_preds = rf_cv.best_estimator_.predict(X_val)
rf_val_scores = get_test_scores('Random Forest (val)', rf_val_preds, y_val)
print("Random Forest results on validation set:")
print(rf_val_scores)

# اطبعلي confusion_matrix للراندوم فورست

cm_rf = confusion_matrix(y_val, rf_val_preds, labels=rf_cv.classes_)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Retained', 'Churned'])
disp_rf.plot(cmap='Blues')
plt.title('Confusion Matrix - Random Forest Model on Validation Data')
plt.savefig(r'C:/Users/duaar/OneDrive/Desktop/image/Confusion_Matrix_-_Random_Forest_Model_on_Validation_Data.png')
plt.show()


# --- 6.3. XGBoost Model ---
print("\n--- Training XGBoost Model ---")
xgb = XGBClassifier(objective='binary:logistic', random_state=42)
cv_params_xgb = {'max_depth': [6, 12], 'min_child_weight': [3, 5],
                 'learning_rate': [0.01, 0.1], 'n_estimators': [300]}
xgb_cv = GridSearchCV(xgb, cv_params_xgb, scoring=scoring, cv=4, refit='recall')
xgb_cv.fit(X_train, y_train)

print("Best parameters for XGBoost:", xgb_cv.best_params_)

# Evaluate on validation set
xgb_val_preds = xgb_cv.best_estimator_.predict(X_val)
xgb_val_scores = get_test_scores('XGBoost (val)', xgb_val_preds, y_val)
print("XGBoost results on validation set:")
print(xgb_val_scores)


# ==================================================================================
# # 7. Final Model Evaluation and Selection
# ==================================================================================
print("\n\n# ==================================================================================")
print("# 7. Final Model Evaluation and Selection")
print("# ==================================================================================\n")

# Aggregate validation results
results = pd.concat([rf_val_scores, xgb_val_scores], axis=0)
print("### Comparing model performance on the validation set:")
print(results)

# Based on the results, XGBoost appears to be the best model (as it often is).
# We will evaluate it on the final test set.
print("\n### Evaluating the final XGBoost model on the test set:")
xgb_test_preds = xgb_cv.best_estimator_.predict(X_test)
xgb_test_scores = get_test_scores('XGBoost (test)', xgb_test_preds, y_test)
print(xgb_test_scores)

# --- 7.1. Confusion Matrix for the Final Model ---
cm = confusion_matrix(y_test, xgb_test_preds, labels=xgb_cv.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Retained', 'Churned'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - XGBoost Model on Test Data')
plt.savefig(r'C:/Users/duaar/OneDrive/Desktop/image/Confusion_Matrix_-_XGBoost_Model_on_Test_Data.png')
plt.show()

print("\n### Detailed Classification Report for XGBoost Model:")
print(classification_report(y_test, xgb_test_preds, target_names=['Retained', 'Churned']))

# --- 7.2. Feature Importance ---
plt.figure(figsize=(10, 8))
plot_importance(xgb_cv.best_estimator_, max_num_features=10)
plt.title('Top 10 Features in XGBoost Model')
plt.savefig(r'C:/Users/duaar/OneDrive/Desktop/image/Top_10_Features_in_XGBoost_Model.png')
plt.show()


# ==================================================================================
# # 8. Threshold Tuning
# ==================================================================================
print("\n\n# ==================================================================================")
print("# 8. Threshold Tuning")
print("# ==================================================================================\n")

def threshold_finder(y_true, probabilities, desired_recall):
    '''
    Finds the best decision threshold to achieve a specific recall value.
    '''
    probs = [x[1] for x in probabilities]
    thresholds = np.arange(0, 1, 0.001)
    scores = []
    for threshold in thresholds:
        preds = np.array([1 if x >= threshold else 0 for x in probs])
        recall = recall_score(y_true, preds)
        scores.append((threshold, recall))
    distances = [abs(score[1] - desired_recall) for score in scores]
    best_idx = np.argmin(distances)
    return scores[best_idx]

# Get prediction probabilities from the XGBoost model
predicted_probabilities = xgb_cv.best_estimator_.predict_proba(X_test)

# Find the threshold that gives a recall of approximately 0.5
best_threshold, best_recall = threshold_finder(y_test, predicted_probabilities, 0.5)
print(f"To achieve a Recall of approximately 0.5, the best threshold is: {best_threshold:.3f} (Actual Recall: {best_recall:.3f})")

# Apply the new threshold
new_preds = np.array([1 if x[1] >= best_threshold else 0 for x in predicted_probabilities])

# Evaluate results with the new threshold
xgb_tuned_scores = get_test_scores(f'XGBoost (test, threshold={best_threshold:.3f})', new_preds, y_test)

# Display final results for comparison
final_results = pd.concat([xgb_test_scores, xgb_tuned_scores], axis=0)
print("\n### Comparing final results before and after threshold tuning:")
print(final_results)

print("\n### Classification report after threshold tuning:")
print(classification_report(y_test, new_preds, target_names=['Retained', 'Churned']))


import os


# Define the list of original and new columns
original_columns = [
    'ID', 'label', 'sessions', 'drives', 'total_sessions', 'n_days_after_onboarding',
    'total_navigations_fav1', 'total_navigations_fav2', 'driven_km_drives',
    'duration_minutes_drives', 'activity_days', 'driving_days', 'device'
]
new_columns = [
    'km_per_drive', 'km_per_driving_day', 'drives_per_driving_day', 'percent_sessions_in_last_month',
    'professional_driver', 'total_sessions_per_day', 'km_per_hour', 'percent_of_drives_to_favorite', 'device2'
]
all_columns = original_columns + new_columns

# Ensure all columns exist in the DataFrame (fill missing with NaN if needed)
for col in all_columns:
    if col not in df.columns:
        df[col] = np.nan

# Reorder DataFrame columns
df_export = df[all_columns]

# Save as CSV
csv_output_path = r'C:\Users\duaar\OneDrive\Desktop\Waze\WAZE clean.csv'
df_export.to_csv(csv_output_path, index=False)

# Save as Python file
py_output_path = r'C:\Users\duaar\OneDrive\Desktop\Waze\WAZE_clean.py'
with open(py_output_path, 'w', encoding='utf-8') as f:
    f.write(df_export.to_string(index=False))


   