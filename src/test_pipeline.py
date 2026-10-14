import pandas as pd
import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

stores = pd.read_csv('stores data-set.csv')
features = pd.read_csv('Features data set.csv')
sales = pd.read_csv('sales data-set.csv')

features['Date'] = pd.to_datetime(features['Date'], format='%d/%m/%Y')
sales['Date'] = pd.to_datetime(sales['Date'], format='%d/%m/%Y')

df = sales.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
df = df.merge(stores, on='Store', how='left')

for col in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
    df[col] = df[col].fillna(0)

df['Total_MarkDown'] = df['MarkDown1'] + df['MarkDown2'] + df['MarkDown3'] + df['MarkDown4'] + df['MarkDown5']
df['Has_MarkDown'] = (df['Total_MarkDown'] > 0).astype(int)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
df['Day'] = df['Date'].dt.day

thanksgiving_dates = pd.to_datetime(['2010-11-26', '2011-11-25', '2012-11-23'])
christmas_dates = pd.to_datetime(['2010-12-31', '2011-12-30', '2012-12-28'])
superbowl_dates = pd.to_datetime(['2010-02-12', '2011-02-11', '2012-02-10'])
laborday_dates = pd.to_datetime(['2010-09-10', '2011-09-09', '2012-09-07'])

df['Is_Thanksgiving'] = df['Date'].isin(thanksgiving_dates).astype(int)
df['Is_Christmas'] = df['Date'].isin(christmas_dates).astype(int)
df['Is_SuperBowl'] = df['Date'].isin(superbowl_dates).astype(int)
df['Is_LaborDay'] = df['Date'].isin(laborday_dates).astype(int)

df = pd.get_dummies(df, columns=['Type'], drop_first=False)

train = df[df['Date'] < '2012-02-10'].copy()
test = df[df['Date'] >= '2012-02-10'].copy()

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

feature_cols = [
    'Store', 'Dept', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Total_MarkDown', 'Has_MarkDown',
    'Year', 'Month', 'Week', 'Day',
    'IsHoliday', 'Is_Thanksgiving', 'Is_Christmas', 'Is_SuperBowl', 'Is_LaborDay',
    'Type_A', 'Type_B', 'Type_C',
    'Store_Dept_Avg', 'Store_Avg', 'Dept_Avg'
]

X_train = train[feature_cols].copy()
y_train = train['Weekly_Sales'].copy()
X_test = test[feature_cols].copy()
y_test = test['Weekly_Sales'].copy()

X_train['IsHoliday'] = X_train['IsHoliday'].astype(int)
X_test['IsHoliday'] = X_test['IsHoliday'].astype(int)
for t in ['Type_A', 'Type_B', 'Type_C']:
    X_train[t] = X_train[t].astype(int)
    X_test[t] = X_test[t].astype(int)

def wmae(y_true, y_pred, is_holiday):
    weights = np.where(is_holiday == 1, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# 1. Ridge
t0 = time.time()
r = Ridge()
r.fit(X_train, y_train)
p_r = r.predict(X_test)
print(f"Ridge ({time.time()-t0:.1f}s): RMSE={np.sqrt(mean_squared_error(y_test, p_r)):.2f}, MAE={mean_absolute_error(y_test, p_r):.2f}, WMAE={wmae(y_test, p_r, X_test['IsHoliday']):.2f}, R2={r2_score(y_test, p_r):.4f}")

# 2. LightGBM
t0 = time.time()
lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.08, num_leaves=63, random_state=42, verbose=-1, n_jobs=-1)
lgb_model.fit(X_train, y_train)
p_lgb = lgb_model.predict(X_test)
print(f"LightGBM ({time.time()-t0:.1f}s): RMSE={np.sqrt(mean_squared_error(y_test, p_lgb)):.2f}, MAE={mean_absolute_error(y_test, p_lgb):.2f}, WMAE={wmae(y_test, p_lgb, X_test['IsHoliday']):.2f}, R2={r2_score(y_test, p_lgb):.4f}")

# 3. XGBoost
t0 = time.time()
xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
p_xgb = xgb_model.predict(X_test)
print(f"XGBoost ({time.time()-t0:.1f}s): RMSE={np.sqrt(mean_squared_error(y_test, p_xgb)):.2f}, MAE={mean_absolute_error(y_test, p_xgb):.2f}, WMAE={wmae(y_test, p_xgb, X_test['IsHoliday']):.2f}, R2={r2_score(y_test, p_xgb):.4f}")

# 4. Random Forest (sample or 60 trees for speed)
t0 = time.time()
rf_model = RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
p_rf = rf_model.predict(X_test)
print(f"Random Forest ({time.time()-t0:.1f}s): RMSE={np.sqrt(mean_squared_error(y_test, p_rf)):.2f}, MAE={mean_absolute_error(y_test, p_rf):.2f}, WMAE={wmae(y_test, p_rf, X_test['IsHoliday']):.2f}, R2={r2_score(y_test, p_rf):.4f}")
