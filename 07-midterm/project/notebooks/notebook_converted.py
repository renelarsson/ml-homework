#!/usr/bin/env python
# coding: utf-8

# ## Data Preparation and Data Cleaning
# 
# This section includes loading the dataset, handling missing values, and detecting outliers.

# ### Get the data

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_theme(style="whitegrid")


# In[2]:


# Download the Dataset
import os

def download_dataset():
    # Kaggle CLI command to download the "Credit Card Fraud Detection" dataset
    kaggle_command = "kaggle datasets download -d mlg-ulb/creditcardfraud -p ../data/"
    unzip_command = "unzip -o ../data/creditcardfraud.zip -d ../data/"
    # Check if the dataset already exists to avoid re-downloading
    if not os.path.exists("../data/creditcard.csv"):
        # Execute shell commands
        os.system(kaggle_command)
        os.system(unzip_command)

# Ensure the dataset is downloaded
download_dataset()


# In[3]:


# Load the dataset
data = pd.read_csv("../data/creditcard.csv")


# ### Display basic information

# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# ### Detect missing values and outliers

# In[7]:


# Detect outliers using a threshold of [-5, 5]
import numpy as np

# Count values outside the range [-5, 5]
outlier_counts = {}
for column in data.columns[:-1]:  # Exclude the target column
    outlier_counts[column] = ((data[column] > 5) | (data[column] < -5)).sum()

print("Outlier counts for values outside [-5, 5]:")
print(outlier_counts)


# In[8]:


# Cap values within the range [-5, 5]
data_clipped = data.copy()
data_clipped.iloc[:, :-1] = data_clipped.iloc[:, :-1].apply(lambda x: np.clip(x, -5, 5))


# In[9]:


# Apply log transformation to the Amount column
data_clipped['Amount'] = np.log1p(data_clipped['Amount'])  # log1p handles log(0) safely

print("Log-transformed 'Amount' column:")
print(data_clipped['Amount'].describe())


# ## EDA and Feature Importance Analysis
# 
# This section explores the dataset, analyzes class imbalance, and evaluates feature importance using mutual information, logistic regression coefficients, and tree-based methods.

# ### Check target class imbalance

# In[10]:


class_counts = data['Class'].value_counts()
class_counts


# In[11]:


# Target class distribution plot
plt.figure(figsize=(8, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette="viridis", dodge=False)
for i, value in enumerate(class_counts.values):
    plt.text(i, value + 100, f'{(value / class_counts.sum() * 100):.2f}%', ha='center')
plt.title("Target Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()


# ### Check Time feature and correlation matrix

# In[12]:


# Check Time feature plot (the time elapsed between transactions) 
data.Time.hist(bins=30, figsize=(10, 5))
plt.suptitle("Time elapsed between transactions")
plt.show()


# In[13]:


# Feature correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()


# ### Feature importance methods

# In[14]:


# Analyze feature importance using mutual information
from sklearn.feature_selection import mutual_info_classif #, mutual_info_score

X = data.drop(columns=['Class'])
y = data['Class']
mutual_info = mutual_info_classif(X, y)
# Indexing the mutual_info array with pd.Series
mutual_info_series = pd.Series(mutual_info, index=X.columns).sort_values(ascending=False)
print("Top 5 Features by mutual information:")
print(mutual_info_series.head())


# In[15]:


# Analyze feature importance using logistic regression coefficients
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Split the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train a Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=2000)
model.fit(X_train_scaled, y_train)
#model.fit(X_train, y_train)

# Get feature importance from coefficients using the validation set
val_feature_importance = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
print("Feature Importance based on Logistic Regression (Validation Set):")
print(val_feature_importance.head())


# In[16]:


# Analyze feature importance using tree-based feature importance
from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest model with max_depth to prevent overfitting
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importance from the Random Forest model
rf_feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importance based on Random Forest:")
print(rf_feature_importance.head())


# ### Feature Importance Analysis
# 
# Feature importance using three different methods:
# 
# 1. **Mutual Information**:
#    - This method measures the dependency between each feature and the target variable.
#    - Top 5 features by mutual information:
#      - V17, V14, V12, V10, V16
# 
# 2. **Logistic Regression Coefficients**:
#    - This method identifies features that have the most influence on the model's predictions.
#    - Results with `max_iter=2000` and scaling:
#      - Top 5 features: V4, V22, V21, Amount, V5
#    - Scaling was necessary to ensure convergence and reliable results.
# 
# 3. **Tree-Based Feature Importance (Random Forest)**:
#    - This method evaluates the importance of features based on their contribution to reducing impurity in decision trees.
#    - Top 5 features:
#      - V17, V12, V14, V10, V16
# 
# ### Observations:
# - Features `V17`, `V14`, and `V12` consistently appear as important across multiple methods.
# - The `Amount` feature is highlighted by Logistic Regression after scaling, indicating its relevance when properly normalized.
# - Tree-based methods and mutual information emphasize `V17` and `V14`, suggesting their strong predictive power.

# In[17]:


# Summarize feature importance results
feature_importance_summary = {
    "Mutual Information": ["V17", "V14", "V12", "V10", "V16"],
    "Logistic Regression (scaled)": ["V4", "V22", "V21", "Amount", "V5"],
    "Random Forest": ["V17", "V12", "V14", "V10", "V16"]
}
import pandas as pd
importance_df = pd.DataFrame.from_dict(feature_importance_summary, orient="index").transpose()
importance_df.columns = ["Mutual Information", "Logistic Regression (scaled)", "Random Forest"]
print("Feature Importance Summary:")
print(importance_df)


# ## Model Selection Process and Parameter Tuning
# 
# This section will include training different models, performing cross-validation, and tuning hyperparameters to optimize performance.
# 
# ### Considerations
# - Use all features for the baseline model to evaluate overall performance.
# - Experiment with a subset of the most important features (`V17`, `V14`, `V12`, `V4`, `Amount`) to reduce dimensionality and improve efficiency.

# In[18]:


# Decide on features for the baseline model
# Option 1: Use all features
X_all_features = data_clipped.drop(columns=['Class'])
y = data_clipped['Class']

# Option 2: Use a subset of the most important features
important_features = ['V17', 'V14', 'V12', 'V4', 'Amount']
X_important_features = data_clipped[important_features]

# Print the shapes of the datasets to verify
print("All features dataset shape:", X_all_features.shape)
print("Important features dataset shape:", X_important_features.shape)


# In[19]:


# Train and evaluate baseline models using both feature sets
from sklearn.metrics import classification_report, roc_auc_score

# Since we're evaluating baseline models, a simple train-test split is sufficient for an initial comparison
X_train_all, X_test_all, y_train, y_test = train_test_split(X_all_features, y, test_size=0.2, random_state=42)
X_train_important, X_test_important, _, _ = train_test_split(X_important_features, y, test_size=0.2, random_state=42)


# In[20]:


# Train and evaluate using all features
model_all = LogisticRegression(random_state=42, max_iter=2000)
model_all.fit(X_train_all, y_train)
y_pred_all = model_all.predict(X_test_all)
y_proba_all = model_all.predict_proba(X_test_all)[:, 1]
print("Performance using all features:")
print(classification_report(y_test, y_pred_all))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba_all))


# In[21]:


# Train and evaluate using important features
model_important = LogisticRegression(random_state=42, max_iter=2000)
model_important.fit(X_train_important, y_train)
y_pred_important = model_important.predict(X_test_important)
y_proba_important = model_important.predict_proba(X_test_important)[:, 1]
print("\nPerformance using important features:")
print(classification_report(y_test, y_pred_important))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba_important))


# In[22]:


# Implement the baseline model using the important features subset
X_train, X_temp, y_train, y_temp = train_test_split(X_important_features, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the Logistic Regression model
baseline_model = LogisticRegression(random_state=42, max_iter=2000)
baseline_model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_pred = baseline_model.predict(X_val)
y_val_proba = baseline_model.predict_proba(X_val)[:, 1]

print("Validation Set Performance:")
print(classification_report(y_val, y_val_pred))
print("ROC AUC Score (Validation):", roc_auc_score(y_val, y_val_proba))


# From the validation set performance metrics, we can interpret the following:
# 
# ### Observations:
# 
# 1. **Class 0 (Non-Fraudulent Transactions):** Precision, Recall, F1-Score: All are 1.00, indicating perfect classification for the majority class. This is expected due to the class imbalance, as the model is highly confident in identifying non-fraudulent transactions.
# 
# 2. **Class 1 (Fraudulent Transactions):**
# - Precision: 0.89, meaning 89% of the transactions predicted as fraudulent are actually fraudulent.
# - Recall: 0.63, meaning the model correctly identifies 63% of the actual fraudulent transactions. This indicates that some fraudulent transactions are being missed.
# - F1-Score: 0.74, which balances precision and recall. This is a reasonable score given the class imbalance.
# 
# 3. **Overall Accuracy:** The accuracy is 1.00, but this is heavily influenced by the majority class (Class 0). Accuracy is not a reliable metric in imbalanced datasets.
# 
# 4. **Macro Average:** Precision, Recall, F1-Score: These metrics average the scores for both classes equally, regardless of class imbalance. The recall of 0.82 highlights the model's difficulty in identifying all fraudulent transactions.
# 
# 5. **Weighted Average:** These metrics account for class imbalance, giving more weight to the majority class. The weighted averages are close to 1.00 due to the dominance of Class 0.
# 
# 6. **ROC AUC Score:** The ROC AUC score of 0.977 indicates excellent discrimination between the classes. This suggests the model is effective at ranking fraudulent transactions higher than non-fraudulent ones.
# 
# ### Interpretation:
# 
# - The model performs very well overall, with a high ROC AUC score and strong precision for fraudulent transactions.
# - However, the recall for Class 1 (fraudulent transactions) is relatively low (0.63), meaning the -model misses some fraudulent transactions. This is a common challenge in imbalanced datasets.
# - The high precision for Class 1 indicates that when the model predicts a transaction as fraudulent, it is likely correct.

# In[ ]:


#!pip install imbalanced-learn


# In[25]:


# Address class imbalance using SMOTE
from imblearn.over_sampling import SMOTE

# SMOTE generates synthetic samples for the minority class to balance the dataset.
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train the Logistic Regression model on the balanced dataset
baseline_model_smote = LogisticRegression(random_state=42, max_iter=2000)
baseline_model_smote.fit(X_train_smote, y_train_smote)

# Evaluate the model on the validation set
y_val_pred_smote = baseline_model_smote.predict(X_val)
y_val_proba_smote = baseline_model_smote.predict_proba(X_val)[:, 1]

print("Validation Set Performance (After SMOTE):")
print(classification_report(y_val, y_val_pred_smote))
print("ROC AUC Score (Validation, After SMOTE):", roc_auc_score(y_val, y_val_proba_smote))


# ### Interpretation:
# - **Improved Recall for Class 1:** The recall for fraudulent transactions (Class 1) has significantly improved from 0.63 to 0.90, meaning the model is now identifying most fraudulent transactions.
# - **Trade-off in Precision:** The precision for Class 1 has dropped to 0.05, meaning many non-fraudulent transactions are being misclassified as fraudulent. This is a common trade-off when addressing class imbalance.
# 
# ### Threshold Adjustment:
# In imbalanced datasets like fraud detection, the default threshold of 0.5 may not be optimal:
# - Fraudulent transactions (Class 1) are rare, so the model might prioritize the majority class (Class 0) at the default threshold.
# - Adjusting the threshold allows us to better handle the trade-off between false positives and false negatives, which is critical in real-world applications.
# 
# In binary classification, models like Logistic Regression output predicted probabilities for each class. Threshold adjustment allows us to balance precision and recall by changing the decision boundary for classifying a transaction as fraudulent. By default, the threshold is 0.5:
# - If the predicted probability for Class 1 is ≥ 0.5, the instance is classified as Class 1.
# - If the predicted probability for Class 1 is < 0.5, the instance is classified as Class 0.
# 
# By lowering or raising the threshold recall or precision can improve:
# - If we want to increase recall (catch more fraudulent transactions), we can lower the threshold (e.g., 0.3). This means the model will classify more instances as Class 1, even if it's less confident.
# - If we want to increase precision (reduce false positives), we can raise the threshold (e.g., 0.7). This means the model will classify fewer instances as Class 1, but with higher confidence.

# In[33]:


# Adjust the classification threshold and evaluate the model
from sklearn.metrics import precision_recall_curve

# Get predicted validation probabilities to allow us to adjust the threshold to balance precision and recall
y_val_proba = baseline_model_smote.predict_proba(X_val)[:, 1]

# Define a function to evaluate the model at different thresholds
def evaluate_threshold(threshold):
    y_val_adjusted = (y_val_proba >= threshold).astype(int)
    print(f"Performance at Threshold = {threshold:.2f}")
    print(classification_report(y_val, y_val_adjusted))
    print("ROC AUC Score:", roc_auc_score(y_val, y_val_adjusted))
    print("-" * 50)

# Evaluate the model at different thresholds
thresholds = [0.7, 0.8, 0.9, 0.91, 0.93, 0.95, 0.97, 0.99]
for threshold in thresholds:
    evaluate_threshold(threshold)


# In[34]:


# Plot the precision-recall curve to visualize the trade-off
precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()


# ### Threshold Selection Rationale
# 
# After evaluating the model's performance at various thresholds, we selected **Threshold = 0.95** for the following reasons:
# 
# 1. **Balanced Trade-Off**:
#     - Precision: 0.44
#     - Recall: 0.83
#     - F1-Score: 0.58
#     - This threshold provides a strong balance between precision and recall, ensuring that most fraudulent transactions are identified while minimizing false positives.
# 2. **Business Alignment**: The selected threshold aligns with general business requirement to reduce false positives (customer inconvenience) while maintaining high recall to catch fraudulent transactions.
# 3. **Performance Metrics**:
#     - Macro Average:
#         - Precision: 0.72
#         - Recall: 0.91
#         - F1-Score: 0.79
#     - Weighted Average:
#         - Precision: 1.00
#         - Recall: 1.00
#         - F1-Score: 1.00
#     - These metrics indicate strong overall performance across both classes.
# 
# ### Class Weights:
# 
# class_weight={0: 1, 1: 10} assigns a higher weight to the minority class (Class 1). This makes the model more sensitive to fraudulent transactions.

# In[42]:


from sklearn.metrics import precision_score, recall_score, f1_score

results = []
for w in range(40, 61, 1):
    # Assign higher weight to the minority class (fraudulent transactions)
    class_weights = {0: 1, 1: w}
    weighted_model = LogisticRegression(random_state=42, max_iter=2000, class_weight=class_weights)

    # Train the model on the original training set
    weighted_model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_val_proba_weighted = weighted_model.predict_proba(X_val)[:, 1]
    y_val_adjusted_weighted = (y_val_proba_weighted >= 0.95).astype(int)

    # Store results
    results.append({
        "class_weight": w,
        "precision": precision_score(y_val, y_val_adjusted_weighted),
        "recall": recall_score(y_val, y_val_adjusted_weighted),
        "f1_score": f1_score(y_val, y_val_adjusted_weighted),
        "roc_auc": roc_auc_score(y_val, y_val_proba_weighted)
    })

# Print results
for result in results:
    print(result)


# In[43]:


# Fine-tune the threshold for class_weight=50
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import numpy as np

# Train the model with class_weight=50
class_weights = {0: 1, 1: 50}
weighted_model = LogisticRegression(random_state=42, max_iter=2000, class_weight=class_weights)
weighted_model.fit(X_train, y_train)

# Get predicted probabilities for the validation set
y_val_proba_weighted = weighted_model.predict_proba(X_val)[:, 1]

# Define thresholds to test
thresholds = np.arange(0.90, 1.00, 0.01)

# Evaluate the model at each threshold
results = []
for threshold in thresholds:
    y_val_adjusted = (y_val_proba_weighted >= threshold).astype(int)
    precision = precision_score(y_val, y_val_adjusted)
    recall = recall_score(y_val, y_val_adjusted)
    f1 = f1_score(y_val, y_val_adjusted)
    roc_auc = roc_auc_score(y_val, y_val_proba_weighted)
    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })

# Print results
for result in results:
    print(f"Threshold: {result['threshold']:.2f}, Precision: {result['precision']:.2f}, Recall: {result['recall']:.2f}, F1-Score: {result['f1_score']:.2f}, ROC AUC: {result['roc_auc']:.2f}")


# In[44]:


# Plot Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba_weighted)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()


# ### Decision on Class Weight and Threshold
# 
# After evaluation, the following configuration was selected as the optimal setup for the baseline model:
# - **Class Weight**: 50
# - **Threshold**: 0.92
# 
# #### Rationale
# 1. **Class Weight = 50**:
#    - Assigning a higher weight to the minority class (fraudulent transactions) during training improved the model's sensitivity to fraud detection.
#    - This resulted in a significant improvement in recall while maintaining high precision.
# 
# 2. **Threshold = 0.92**:
#    - Fine-tuning the threshold after training allowed for further optimization of the trade-off between precision and recall.
#    - This threshold provided the highest F1-score (0.80), indicating the best balance between precision and recall.
# 
# #### Performance Metrics
# - **Validation Set**:
#   - Precision: 0.84
#   - Recall: 0.76
#   - F1-Score: 0.80
#   - ROC AUC: 0.98
# 
# #### Trade-Offs
# - The selected configuration prioritizes a balance between precision and recall, ensuring that most fraudulent transactions are identified while minimizing false positives.
# - This balance aligns with the business requirement to reduce financial losses from missed fraud while minimizing customer inconvenience.

# In[45]:


# Evaluate the model on the test set
y_test_proba = weighted_model.predict_proba(X_test)[:, 1]
y_test_adjusted = (y_test_proba >= 0.92).astype(int)

# Print performance metrics
print("Test Set Performance (Class Weight = 50, Threshold = 0.92):")
print(classification_report(y_test, y_test_adjusted))
print("ROC AUC Score (Test Set):", roc_auc_score(y_test, y_test_proba))


# ### Evaluation of Test Set Performance:
# **Key Metrics:**
# 
# 1. Class 0 (Non-Fraudulent Transactions):
# - Precision, Recall, F1-Score: All are 1.00, indicating perfect classification for the majority class.
# - This is expected due to the class imbalance, as the model is highly confident in identifying non-fraudulent transactions.
# 
# 2. Class 1 (Fraudulent Transactions):
# - Precision: 0.72
#     - 72% of the transactions predicted as fraudulent are actually fraudulent.
# - Recall: 0.82
#     - 82% of the actual fraudulent transactions are correctly identified.
# - F1-Score: 0.76
#     - This reflects a good balance between precision and recall for the minority class.
# 
# 3. Overall Accuracy:
# - The accuracy is 1.00, but this is heavily influenced by the majority class (Class 0). Accuracy is not a reliable metric for imbalanced datasets.
# 
# 4. Macro Average:
# - Precision: 0.86, Recall: 0.91, F1-Score: 0.88
# - These metrics average the scores for both classes equally, regardless of class imbalance.
# 
# 5. Weighted Average:
# - Precision, Recall, F1-Score: All are 1.00, reflecting the dominance of Class 0 in the dataset.
# 
# 6. ROC AUC Score:
# - The ROC AUC score of 0.982 indicates excellent discrimination between the classes.
# 
# #### Comparison with Validation Set:
# - Validation Set:
#     - Precision: 0.84, Recall: 0.76, F1-Score: 0.80, ROC AUC: 0.98
# - Test Set:
#     - Precision: 0.72, Recall: 0.82, F1-Score: 0.76, ROC AUC: 0.98
# - Observations:
#     - The test set results are consistent with the validation set, with a slight drop in precision but an improvement in recall.
#     - The F1-score is slightly lower on the test set, but the ROC AUC score remains excellent, indicating strong generalization.

# In[ ]:


# Save the trained model to a file
import joblib

# Define the file path for saving the model
model_file_path = "../service/baseline_model.pkl"

# Save the model using joblib
joblib.dump(baseline_model, model_file_path)
print(f"Model saved to {model_file_path}")


# In[ ]:




