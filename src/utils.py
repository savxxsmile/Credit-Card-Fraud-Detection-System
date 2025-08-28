# src/utils.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Handle imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    
    return X_res, y_res, scaler

def save_artifact(obj, path):
    joblib.dump(obj, path)
