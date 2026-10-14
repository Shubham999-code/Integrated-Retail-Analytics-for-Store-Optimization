"""
data_preprocessing.py
---------------------
Data cleaning, integration, feature engineering, and splitting
for the Integrated Retail Analytics project.
"""

import pandas as pd
import numpy as np

def load_and_merge_data(stores_path='stores data-set.csv',
                        features_path='Features data set.csv',
                        sales_path='sales data-set.csv'):
    """
    Loads raw CSV files, parses dates, and merges them into a unified DataFrame.
    """
    stores = pd.read_csv(stores_path)
    features = pd.read_csv(features_path)
    sales = pd.read_csv(sales_path)

    features['Date'] = pd.to_datetime(features['Date'], format='%d/%m/%Y')
    sales['Date'] = pd.to_datetime(sales['Date'], format='%d/%m/%Y')

    # Merge sales with features on Store, Date, IsHoliday
    df = sales.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
    # Merge with stores on Store
    df = df.merge(stores, on='Store', how='left')
    
    # Sort chronologically and by store/dept
    df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
    return df

def clean_and_impute_data(df):
    """
    Handles missing values and clean data irregularities:
    - MarkDown1 to MarkDown5: NaN filled with 0 (promotions not running/recorded before Nov 2011)
    - Returns/Negative sales: Separated or flagged for reverse logistics tracking
    """
    df = df.copy()
    
    # MarkDown null imputation
    md_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in md_cols:
        df[col] = df[col].fillna(0)
        
    df['Total_MarkDown'] = df[md_cols].sum(axis=1)
    df['Has_MarkDown'] = (df['Total_MarkDown'] > 0).astype(int)
    
    # Flag negative sales (Returns)
    df['Is_Return'] = (df['Weekly_Sales'] < 0).astype(int)
    
    # Fill CPI and Unemployment if any nulls exist using forward/backward fill
    df['CPI'] = df.groupby('Store')['CPI'].transform(lambda x: x.ffill().bfill())
    df['Unemployment'] = df.groupby('Store')['Unemployment'].transform(lambda x: x.ffill().bfill())
    
    return df

def engineer_features(df):
    """
    Engineers rich temporal, holiday-specific, store-level, and economic features.
    """
    df = df.copy()
    
    # Date decompositions
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
    
    # Specific holiday alignments in the Walmart calendar
    thanksgiving_dates = pd.to_datetime(['2010-11-26', '2011-11-25', '2012-11-23'])
    christmas_dates = pd.to_datetime(['2010-12-31', '2011-12-30', '2012-12-28'])
    superbowl_dates = pd.to_datetime(['2010-02-12', '2011-02-11', '2012-02-10'])
    laborday_dates = pd.to_datetime(['2010-09-10', '2011-09-09', '2012-09-07'])
    
    # Holiday indicators
    df['Is_Thanksgiving'] = df['Date'].isin(thanksgiving_dates).astype(int)
    df['Is_Christmas'] = df['Date'].isin(christmas_dates).astype(int)
    df['Is_SuperBowl'] = df['Date'].isin(superbowl_dates).astype(int)
    df['Is_LaborDay'] = df['Date'].isin(laborday_dates).astype(int)
    
    # Week before major holidays (Pre-holiday shopping surge)
    pre_thanksgiving = thanksgiving_dates - pd.Timedelta(days=7)
    pre_christmas = christmas_dates - pd.Timedelta(days=7)
    df['Is_Pre_Holiday'] = (df['Date'].isin(pre_thanksgiving) | df['Date'].isin(pre_christmas)).astype(int)
    
    # Store Type One-Hot Encoding
    df = pd.get_dummies(df, columns=['Type'], drop_first=False)
    for col in ['Type_A', 'Type_B', 'Type_C']:
        if col in df.columns:
            df[col] = df[col].astype(int)
            
    # Economic interaction features
    df['Fuel_to_CPI_Ratio'] = df['Fuel_Price'] / (df['CPI'] + 1e-5)
    df['Sales_Efficiency_Potential'] = df['Size'] / 1000.0
    
    return df

def split_train_test(df, split_date='2012-02-10'):
    """
    Performs out-of-time train/test split to avoid lookahead leakage.
    Target encodings are strictly computed on the training partition.
    """
    train = df[df['Date'] < split_date].copy()
    test = df[df['Date'] >= split_date].copy()
    
    # Compute historical target encodings solely on train set
    store_dept_avg = train.groupby(['Store', 'Dept'])['Weekly_Sales'].mean().rename('Store_Dept_Avg')
    store_avg = train.groupby('Store')['Weekly_Sales'].mean().rename('Store_Avg')
    dept_avg = train.groupby('Dept')['Weekly_Sales'].mean().rename('Dept_Avg')
    
    train = train.merge(store_dept_avg, on=['Store', 'Dept'], how='left')
    train = train.merge(store_avg, on='Store', how='left')
    train = train.merge(dept_avg, on='Dept', how='left')
    
    test = test.merge(store_dept_avg, on=['Store', 'Dept'], how='left')
    test = test.merge(store_avg, on='Store', how='left')
    test = test.merge(dept_avg, on='Dept', how='left')
    
    overall_mean = train['Weekly_Sales'].mean()
    test['Store_Dept_Avg'] = test['Store_Dept_Avg'].fillna(test['Dept_Avg']).fillna(overall_mean)
    test['Store_Avg'] = test['Store_Avg'].fillna(overall_mean)
    test['Dept_Avg'] = test['Dept_Avg'].fillna(overall_mean)
    
    return train, test
