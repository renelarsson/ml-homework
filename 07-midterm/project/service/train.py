import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Download the dataset
def download_dataset():
    kaggle_command = "kaggle datasets download -d mlg-ulb/creditcardfraud -p ../data/"
    unzip_command = "unzip -o ../data/creditcardfraud.zip -d ../data/"
    if not os.path.exists("../data/creditcard.csv"):
        os.system(kaggle_command)
        os.system(unzip_command)

# Load and preprocess the dataset
def load_data():
    df = pd.read_csv("../data/creditcard.csv")

    # Cap outliers
    df.iloc[:, :-1] = df.iloc[:, :-1].apply(lambda x: np.clip(x, -5, 5))

    # Log-transform the Amount column
    df['Amount'] = np.log1p(df['Amount'])

    features = ['V17', 'V14', 'V12', 'V4', 'Amount']
    target = 'Class'
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
def train_model(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(class_weight={0: 1, 1: 50}, random_state=42))
    ])
    pipeline.fit(X_resampled, y_resampled)
    return pipeline

# Save the model
def save_model(model, output_path):
    joblib.dump(model, output_path)

if __name__ == "__main__":
    download_dataset()
    X_train, X_test, y_train, y_test = load_data()

    # Combine training and validation sets for final training
    X_train_full = pd.concat([X_train, X_test])
    y_train_full = pd.concat([y_train, y_test])

    model = train_model(X_train_full, y_train_full)
    save_model(model, "../project/service/best_model.pkl")
    print("Model training complete. Saved as 'best_model.pkl'.")