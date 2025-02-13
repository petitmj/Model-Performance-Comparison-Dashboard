import os
import pandas as pd
import numpy as np
import boto3
import streamlit as st
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# AWS S3 Configuration
S3_BUCKET_NAME = "arv7staging"
s3_client = boto3.client("s3")

# Function to check if a file exists in S3
def s3_file_exists(s3_key):
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return True
    except:
        return False

# Function to upload local dataset to S3 if not already present
def upload_to_s3(local_path, s3_key):
    if os.path.exists(local_path):
        if not s3_file_exists(s3_key):
            with open(local_path, "rb") as f:
                s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=s3_key, Body=f)
            st.success(f"Uploaded {s3_key} to S3.")
        else:
            st.info(f"Skipping {s3_key}: Already exists in S3.")
    else:
        st.error(f"Error: {local_path} does not exist.")

# Function to load dataset from S3 with support for CSV and XLSX
def load_from_s3(s3_key):
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        if s3_key.endswith(".csv"):
            df = pd.read_csv(obj['Body'], low_memory=False)
        elif s3_key.endswith(".xlsx"):
            df = pd.read_excel(BytesIO(obj['Body'].read()), engine='openpyxl')
        else:
            raise ValueError("Unsupported file format")
        return df
    except Exception as e:
        st.error(f"Error loading {s3_key} from S3: {e}")
        return None

# Function to preprocess data
def preprocess_data(df, target_column):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    
    df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
    if target_column not in df.columns:
        raise ValueError(f"After processing, target column '{target_column}' is missing.")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Available models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Support Vector Machine": SVC(probability=True),
    "HistGradientBoosting": HistGradientBoostingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Streamlit UI
st.title("AI ElderCare Dashboard")
uploaded_file = st.file_uploader("Upload CSV/XLSX File", type=["csv", "xlsx"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1]
    df = pd.read_csv(uploaded_file) if file_extension == "csv" else pd.read_excel(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())
    
    target_column = st.selectbox("Select Target Column", df.columns)
    selected_model = st.selectbox("Select Model", list(models.keys()))
    
    if st.button("Train Model"):
        try:
            X_train, X_test, y_train, y_test = preprocess_data(df, target_column)
            model = models[selected_model]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            results = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "ROC AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"
            }
            
            st.write("### Model Performance")
            st.json(results)
        except Exception as e:
            st.error(f"Error: {e}")
