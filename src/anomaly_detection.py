"""
anomaly_detection.py
--------------------
Anomaly detection in retail sales data using statistical,
machine learning, and time-series decomposition methods.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose

def detect_statistical_anomalies(df, col='Weekly_Sales'):
    """
    Detects anomalies using standard IQR and Z-Score methods.
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    iqr_anomalies = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    mean_val = df[col].mean()
    std_val = df[col].std()
    z_scores = np.abs((df[col] - mean_val) / std_val)
    z_anomalies = z_scores > 3.0
    
    return {
        'iqr_mask': iqr_anomalies,
        'iqr_bounds': (lower_bound, upper_bound),
        'z_mask': z_anomalies,
        'z_scores': z_scores
    }

def detect_isolation_forest_anomalies(store_week_df, contamination=0.03, random_state=42):
    """
    Fits an Isolation Forest model on multi-dimensional store-week metrics.
    Features: Weekly_Sales, Total_MarkDown, Temperature, Fuel_Price, CPI, Unemployment, Size.
    """
    feature_cols = ['Weekly_Sales', 'Total_MarkDown', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']
    X = store_week_df[feature_cols].copy()
    
    iso_model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    preds = iso_model.fit_predict(X)
    scores = iso_model.decision_function(X)
    
    df_result = store_week_df.copy()
    df_result['Iso_Anomaly'] = (preds == -1).astype(int)
    df_result['Anomaly_Score'] = scores
    
    return df_result, iso_model

def detect_time_series_anomalies(df, freq='W'):
    """
    Aggregates overall chain sales by date, performs seasonal decomposition,
    and flags residuals exceeding 2.5 standard deviations as time-based anomalies.
    """
    chain_sales = df.groupby('Date')['Weekly_Sales'].sum().asfreq('W-FRI')
    chain_sales = chain_sales.interpolate(method='linear')
    
    # Seasonal decomposition (period=52 weeks in a year)
    decomp = seasonal_decompose(chain_sales, model='additive', period=52)
    resid = decomp.resid.dropna()
    
    mean_resid = resid.mean()
    std_resid = resid.std()
    ts_anomalies = resid[np.abs(resid - mean_resid) > 2.5 * std_resid]
    
    return {
        'decomposition': decomp,
        'residuals': resid,
        'anomalous_dates': ts_anomalies.index,
        'anomalous_values': ts_anomalies.values
    }

def categorize_anomalies(store_week_df):
    """
    Categorizes detected anomalies into business causes:
    - Holiday Spike (Thanksgiving, Christmas surge)
    - Promotional / Markdown Surge (Heavy markdown lift)
    - Macroeconomic / Weather Shock (Extreme cold/heat or fuel shock)
    - Operational / Negative Sales (Stockouts, inventory adjustment, or returns)
    """
    df = store_week_df.copy()
    categories = []
    for _, row in df.iterrows():
        if row.get('Iso_Anomaly', 0) == 1:
            if row['IsHoliday']:
                categories.append('Holiday Surge')
            elif row['Total_MarkDown'] > df['Total_MarkDown'].quantile(0.85):
                categories.append('Promotional Lift')
            elif row['Weekly_Sales'] < df['Weekly_Sales'].quantile(0.05):
                categories.append('Low Sales / Operational Disruption')
            elif row['Temperature'] < 25 or row['Temperature'] > 95:
                categories.append('Extreme Weather Impact')
            else:
                categories.append('Unexplained Volume Surge')
        else:
            categories.append('Normal')
            
    df['Anomaly_Category'] = categories
    return df
