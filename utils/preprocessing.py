import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(df):
    """Preprocess solar energy data"""
    df = df.copy()
    
    # Handle missing values
    df = df.ffill()
    df = df.bfill()
    
    # Extract temporal features if datetime column exists
    if 'datetime' in df.columns or 'date' in df.columns:
        date_col = 'datetime' if 'datetime' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col])
        df['hour'] = df[date_col].dt.hour
        df['day'] = df[date_col].dt.day
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.dayofweek
    
    return df

def engineer_features(df, target_col='power_generated'):
    """Create additional features"""
    df = df.copy()
    
    # Rolling averages if enough data and target column exists
    if target_col in df.columns and len(df) > 24:
        df['power_rolling_24h'] = df[target_col].rolling(window=24, min_periods=1).mean()
    
    return df

def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
