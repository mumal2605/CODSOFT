# churn_model.py (Adapted for Bank Customer Churn Dataset)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# --- 1. Configuration & Setup ---
# Assuming your file is named 'churn_data.csv' in the 'data' folder
DATA_PATH = 'data/churn_data.csv'
MODEL_DIR = '.'
PLOTS_DIR = 'plots'
MODEL_NAME = 'bank_churn_model.pkl' # Renamed for clarity
PREPROCESSOR_NAME = 'bank_churn_preprocessor.pkl' # Renamed for clarity

# Create directories if they don't exist
os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_style('whitegrid')

# --- 2. Data Loading and Cleaning ---
def load_and_clean_data(path):
    """Loads data and performs initial cleaning for the Bank Churn dataset."""
    print("Loading and cleaning data...")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {path}")
        return None
    
    # Drop columns that are identifiers or not useful for prediction
    # 'RowNumber', 'CustomerId', 'Surname' are just identifiers.
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    
    print("Data loaded and cleaned successfully.")
    return df

# --- 3. Exploratory Data Analysis (EDA) ---
def perform_eda(df):
    """Performs EDA and saves plots for the Bank Churn dataset."""
    print("Performing Exploratory Data Analysis...")
    
    # Target variable is 'Exited' (1 = Churned, 0 = Stayed)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Exited', data=df.assign(Exited=df['Exited'].map({1: 'Exited', 0: 'Stayed'})))
    plt.title('Churn (Exited) Distribution')
    plt.savefig(os.path.join(PLOTS_DIR, 'churn_distribution.png'))
    plt.close()

    # Churn vs. Categorical Features
    categorical_features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, hue=df['Exited'].map({1: 'Exited', 0: 'Stayed'}), data=df)
        plt.title(f'Churn by {feature}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'churn_by_{feature}.png'))
        plt.close()

    # Correlation Heatmap for numerical features
    plt.figure(figsize=(12, 8))
    numerical_cols = df.select_dtypes(include=np.number).columns
    sns.heatmap(df[numerical_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
    plt.close()
    
    print("EDA plots saved to 'plots/' directory.")

# --- 4. Data Preprocessing ---
def preprocess_data(df):
    """Preprocesses data: splits, scales, encodes, and handles imbalance."""
    print("Preprocessing data...")
    # The target variable 'Exited' is our 'y'
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Handle imbalance with SMOTE (since churn is usually imbalanced)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
    
    print("Data preprocessing complete.")
    return X_train_smote, y_train_smote, X_test_processed, y_test, preprocessor

# --- 5. Model Training and Evaluation ---
def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Trains and evaluates multiple models, returns the best one."""
    print("Training and evaluating models...")
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    best_model = None
    best_roc_auc = -1

    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred))
        results[name] = roc_auc
        
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = model
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curves.png'))
    plt.close()
    
    print(f"\nBest model selected: {type(best_model).__name__} with ROC-AUC: {best_roc_auc:.4f}")
    return best_model

# --- 6. Main Execution Block ---
if __name__ == '__main__':
    # Step 1: Load and clean data
    df_cleaned = load_and_clean_data(DATA_PATH)
    
    if df_cleaned is not None:
        # Step 2: Perform EDA
        perform_eda(df_cleaned.copy())
        
        # Step 3: Preprocess data
        X_train_final, y_train_final, X_test_final, y_test_final, preprocessor = preprocess_data(df_cleaned)
        
        # Step 4: Train and find the best model
        final_model = train_and_evaluate(X_train_final, y_train_final, X_test_final, y_test_final)
        
        # Step 5: Save the model and preprocessor
        joblib.dump(final_model, os.path.join(MODEL_DIR, MODEL_NAME))
        joblib.dump(preprocessor, os.path.join(MODEL_DIR, PREPROCESSOR_NAME))
        print(f"\nModel saved as '{MODEL_NAME}'")
        print(f"Preprocessor saved as '{PREPROCESSOR_NAME}'")
        
        print("\n--- Project Execution Complete ---")