"""
forecasting.py
--------------
Demand forecasting models for retail sales:
Linear/Ridge, Random Forest, XGBoost, LightGBM, and SARIMAX.
Evaluates using RMSE, MAE, R², and Walmart Official WMAE (Holiday weights = 5).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def calculate_wmae(y_true, y_pred, is_holiday):
    """
    Computes the official Walmart Weighted Mean Absolute Error (WMAE).
    Weights holiday weeks 5x compared to standard weeks.
    """
    weights = np.where(is_holiday == 1, 5.0, 1.0)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

def evaluate_predictions(y_true, y_pred, is_holiday):
    """
    Computes RMSE, MAE, WMAE, and R2 score.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    wmae = calculate_wmae(y_true, y_pred, is_holiday)
    r2 = r2_score(y_true, y_pred)
    return {
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'WMAE': round(wmae, 2),
        'R2': round(r2, 4)
    }

def train_forecasting_models(X_train, y_train, X_test, y_test):
    """
    Trains Ridge, Random Forest, LightGBM, and XGBoost regressors.
    Returns dictionary of trained models and evaluation metrics.
    """
    models = {}
    metrics = {}
    
    # 1. Ridge Baseline
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train, y_train)
    p_ridge = ridge.predict(X_test)
    models['Ridge'] = ridge
    metrics['Ridge'] = evaluate_predictions(y_test, p_ridge, X_test['IsHoliday'])
    
    # 2. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=60, max_depth=18, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    p_rf = rf.predict(X_test)
    models['Random Forest'] = rf
    metrics['Random Forest'] = evaluate_predictions(y_test, p_rf, X_test['IsHoliday'])
    
    # 3. LightGBM Regressor
    lgbm = lgb.LGBMRegressor(
        n_estimators=220,
        learning_rate=0.07,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm.fit(X_train, y_train)
    p_lgbm = lgbm.predict(X_test)
    models['LightGBM'] = lgbm
    metrics['LightGBM'] = evaluate_predictions(y_test, p_lgbm, X_test['IsHoliday'])
    
    # 4. XGBoost Regressor
    xgbr = xgb.XGBRegressor(
        n_estimators=180,
        learning_rate=0.08,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    xgbr.fit(X_train, y_train)
    p_xgb = xgbr.predict(X_test)
    models['XGBoost'] = xgbr
    metrics['XGBoost'] = evaluate_predictions(y_test, p_xgb, X_test['IsHoliday'])
    
    return models, metrics, {
        'Ridge': p_ridge,
        'Random Forest': p_rf,
        'LightGBM': p_lgbm,
        'XGBoost': p_xgb
    }

def fit_sarimax_chain_baseline(df):
    """
    Fits a SARIMAX time series model on chain-wide weekly sales.
    """
    weekly_agg = df.groupby('Date')['Weekly_Sales'].sum().asfreq('W-FRI')
    train_agg = weekly_agg[:'2012-02-10']
    test_agg = weekly_agg['2012-02-10':]
    
    # Fit SARIMAX(1, 1, 1) x (1, 1, 0, 52) or simple seasonal
    try:
        model = SARIMAX(train_agg, order=(1, 1, 1), seasonal_order=(0, 1, 1, 52), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        preds = res.get_forecast(steps=len(test_agg)).predicted_mean
        rmse = np.sqrt(mean_squared_error(test_agg, preds))
        mae = mean_absolute_error(test_agg, preds)
        return res, preds, {'RMSE': round(rmse, 2), 'MAE': round(mae, 2)}
    except Exception as e:
        # Fallback to non-seasonal ARIMA(1, 1, 1)
        model = SARIMAX(train_agg, order=(1, 1, 1))
        res = model.fit(disp=False)
        preds = res.get_forecast(steps=len(test_agg)).predicted_mean
        rmse = np.sqrt(mean_squared_error(test_agg, preds))
        mae = mean_absolute_error(test_agg, preds)
        return res, preds, {'RMSE': round(rmse, 2), 'MAE': round(mae, 2)}
