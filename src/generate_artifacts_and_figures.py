"""
generate_artifacts_and_figures.py
---------------------------------
Executes the complete retail analytics pipeline, saves high-res charts to reports/figures/,
trains and evaluates all models, and serializes trained models to models/.
"""

import os
import time
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib

# Import local modules
from data_preprocessing import load_and_merge_data, clean_and_impute_data, engineer_features, split_train_test
from anomaly_detection import detect_statistical_anomalies, detect_isolation_forest_anomalies, detect_time_series_anomalies, categorize_anomalies
from segmentation import build_store_profiles, perform_clustering
from market_basket import construct_department_baskets, mine_association_rules, extract_merchandising_recommendations
from forecasting import calculate_wmae, evaluate_predictions, train_forecasting_models, fit_sarimax_chain_baseline

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150

FIG_DIR = 'reports/figures'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

print("="*60)
print("STARTING FULL PIPELINE & ARTIFACT GENERATION")
print("="*60, flush=True)

# 1. Load and Preprocess
print("\n[1/6] Ingesting and Preprocessing Data...", flush=True)
df_raw = load_and_merge_data()
df_cleaned = clean_and_impute_data(df_raw)
df = engineer_features(df_cleaned)
print(f"Data ready: {df.shape[0]} rows, {df.shape[1]} columns", flush=True)

# -------------------------------------------------------------
# 2. EDA & Visualizations
# -------------------------------------------------------------
print("\n[2/6] Generating 15 EDA Figures...", flush=True)

# Chart 1: Total Weekly Sales Trend Over Time
plt.figure(figsize=(14, 5))
chain_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
plt.plot(chain_sales['Date'], chain_sales['Weekly_Sales'] / 1e6, color='#1f77b4', lw=2.2, label='Total Weekly Sales ($M)')
# Highlight Holidays
holidays = df[df['IsHoliday']][['Date', 'Is_Thanksgiving', 'Is_Christmas', 'Is_SuperBowl', 'Is_LaborDay']].drop_duplicates()
for _, row in holidays.iterrows():
    h_label = 'Thanksgiving' if row['Is_Thanksgiving'] else ('Christmas' if row['Is_Christmas'] else ('Super Bowl' if row['Is_SuperBowl'] else 'Labor Day'))
    plt.axvline(row['Date'], color='red' if 'Thanks' in h_label or 'Christ' in h_label else 'orange', linestyle='--', alpha=0.6)
plt.title('Chart 1: Total Chain Weekly Sales Trend (2010 - 2012) with Major Holiday Peaks', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Date', fontsize=11)
plt.ylabel('Weekly Sales ($ Millions)', fontsize=11)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart01_weekly_sales_trend.png')
plt.close()

# Chart 2: Store Type vs Sales & Size
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=df, x='Type_A', y='Weekly_Sales', ax=axes[0], palette='Blues', showfliers=False)
# Grouped by original type
store_type_summary = df.groupby('Store').agg({'Size': 'first', 'Weekly_Sales': 'mean'}).reset_index()
# get original type
stores_meta = pd.read_csv('stores data-set.csv')
store_type_summary = store_type_summary.merge(stores_meta[['Store', 'Type']], on='Store')
sns.boxplot(data=store_type_summary, x='Type', y='Weekly_Sales', ax=axes[0], palette='Blues')
axes[0].set_title('Weekly Sales Distribution by Store Type', fontweight='bold')
axes[0].set_ylabel('Avg Weekly Sales ($)')
sns.barplot(data=store_type_summary, x='Type', y='Size', ax=axes[1], palette='Blues', ci=None)
axes[1].set_title('Average Store Physical Size (Sq. Ft.) by Store Type', fontweight='bold')
axes[1].set_ylabel('Size (Sq. Ft.)')
plt.suptitle('Chart 2: Store Type Performance and Physical Scale Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart02_store_type_comparison.png')
plt.close()

# Chart 3: Top 15 Best-Selling Departments
plt.figure(figsize=(12, 6))
top_depts = df.groupby('Dept')['Weekly_Sales'].sum().sort_values(ascending=False).head(15) / 1e6
sns.barplot(x=top_depts.values, y=[f'Dept {d}' for d in top_depts.index], palette='mako')
plt.title('Chart 3: Top 15 Revenue-Generating Departments (Total Sales in $M)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Total Sales ($ Millions)', fontsize=11)
plt.ylabel('Department', fontsize=11)
for i, v in enumerate(top_depts.values):
    plt.text(v + 1, i, f"${v:.1f}M", va='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart03_top_departments.png')
plt.close()

# Chart 4: Top 10 vs Bottom 10 Stores
plt.figure(figsize=(14, 5))
store_sales = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False) / 1e6
top_bottom = pd.concat([store_sales.head(10), store_sales.tail(10)])
colors = ['#2ca02c']*10 + ['#d62728']*10
plt.bar([f'Store {s}' for s in top_bottom.index], top_bottom.values, color=colors)
plt.title('Chart 4: Benchmark Comparison - Top 10 vs Bottom 10 Performing Stores (Total Sales in $M)', fontsize=13, fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Total Sales ($ Millions)', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart04_top_bottom_stores.png')
plt.close()

# Chart 5: Holiday vs Non-Holiday Sales Distribution
plt.figure(figsize=(10, 5))
sns.kdeplot(df[df['IsHoliday'] == 1]['Weekly_Sales'], label='Holiday Weeks', shade=True, color='crimson', lw=2)
sns.kdeplot(df[df['IsHoliday'] == 0]['Weekly_Sales'], label='Non-Holiday Weeks', shade=True, color='navy', lw=2)
plt.xlim(-5000, 70000)
plt.title('Chart 5: Density Distribution of Weekly Sales - Holiday vs Non-Holiday Weeks', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Weekly Sales ($)', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart05_holiday_distribution.png')
plt.close()

# Chart 6: Seasonal Monthly Sales Variations
plt.figure(figsize=(12, 5))
monthly_trend = df.groupby(['Year', 'Month'])['Weekly_Sales'].mean().unstack(level=0)
for yr in monthly_trend.columns:
    plt.plot(monthly_trend.index, monthly_trend[yr], marker='o', lw=2.2, label=f'Year {yr}')
plt.title('Chart 6: Monthly Average Weekly Sales Trajectory across Calendar Years', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Month (1 = Jan, 12 = Dec)', fontsize=11)
plt.ylabel('Average Weekly Sales ($)', fontsize=11)
plt.xticks(range(1, 13))
plt.legend(title='Calendar Year')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart06_monthly_seasonality.png')
plt.close()

# Chart 7: Impact of MarkDowns on Sales
plt.figure(figsize=(12, 5))
md_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
md_totals = df[md_cols].sum() / 1e6
sns.barplot(x=md_cols, y=md_totals.values, palette='viridis')
plt.title('Chart 7: Total Promotional Investment by MarkDown Program ($ Millions)', fontsize=13, fontweight='bold', pad=12)
plt.ylabel('Promotional Spend ($M)', fontsize=11)
for i, v in enumerate(md_totals.values):
    plt.text(i, v + 0.5, f"${v:.1f}M", ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart07_markdown_distribution.png')
plt.close()

# Chart 8: Fuel Price vs Sales
plt.figure(figsize=(10, 5))
sample_df = df.sample(n=10000, random_state=42)
sns.scatterplot(data=sample_df, x='Fuel_Price', y='Weekly_Sales', hue='Type_A', palette='coolwarm', alpha=0.3)
plt.title('Chart 8: Fuel Price vs Weekly Sales Dispersion', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Fuel Price ($ / Gallon)', fontsize=11)
plt.ylabel('Weekly Sales ($)', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart08_fuel_price_vs_sales.png')
plt.close()

# Chart 9: Unemployment vs Weekly Sales
plt.figure(figsize=(10, 5))
sns.regplot(data=df.sample(n=5000, random_state=42), x='Unemployment', y='Weekly_Sales',
            scatter_kws={'alpha': 0.2, 'color': '#34495e'}, line_kws={'color': 'crimson', 'lw': 2})
plt.title('Chart 9: Unemployment Rate vs Weekly Sales with Linear Trend', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Regional Unemployment Rate (%)', fontsize=11)
plt.ylabel('Weekly Sales ($)', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart09_unemployment_vs_sales.png')
plt.close()

# Chart 10: Temperature vs Sales
plt.figure(figsize=(10, 5))
temp_bins = pd.cut(df['Temperature'], bins=10)
temp_sales = df.groupby(temp_bins)['Weekly_Sales'].mean().reset_index()
temp_sales['Bin_Str'] = temp_sales['Temperature'].astype(str)
sns.barplot(data=temp_sales, x='Bin_Str', y='Weekly_Sales', palette='flare')
plt.title('Chart 10: Average Weekly Sales across Ambient Temperature Ranges (°F)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Temperature Range (°F)', fontsize=11)
plt.ylabel('Average Weekly Sales ($)', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart10_temperature_vs_sales.png')
plt.close()

# Chart 11: CPI vs Sales
plt.figure(figsize=(10, 5))
store_cpi = df.groupby('Date').agg({'CPI': 'mean', 'Weekly_Sales': 'mean'}).reset_index()
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax1.plot(store_cpi['Date'], store_cpi['CPI'], color='darkorange', lw=2, label='Mean CPI')
ax2.plot(store_cpi['Date'], store_cpi['Weekly_Sales'], color='teal', lw=2, label='Avg Weekly Sales')
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Consumer Price Index (CPI)', color='darkorange', fontsize=11)
ax2.set_ylabel('Average Weekly Sales ($)', color='teal', fontsize=11)
plt.title('Chart 11: Temporal Trajectory of CPI Inflation vs Average Weekly Sales', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart11_cpi_vs_sales.png')
plt.close()

# Chart 12: Reverse Logistics / Negative Sales Analysis
plt.figure(figsize=(12, 5))
neg_sales = df[df['Weekly_Sales'] < 0]
top_return_depts = neg_sales.groupby('Dept')['Weekly_Sales'].agg(['count', 'sum']).sort_values('sum').head(10)
sns.barplot(x=top_return_depts['sum'].abs(), y=[f'Dept {d}' for d in top_return_depts.index], palette='Reds_r')
plt.title('Chart 12: Top 10 Departments with Largest Total Return/Refund Dollar Volumes', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Total Negative Sales Dollar Volume ($)', fontsize=11)
plt.ylabel('Department', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart12_negative_sales_returns.png')
plt.close()

# Chart 13: Department Volatility
plt.figure(figsize=(12, 5))
dept_vol = df.groupby('Dept')['Weekly_Sales'].agg(['mean', 'std']).reset_index()
dept_vol['CV'] = dept_vol['std'] / dept_vol['mean']
top_volatile = dept_vol.sort_values('CV', ascending=False).head(15)
sns.barplot(data=top_volatile, x='CV', y=[f'Dept {d}' for d in top_volatile['Dept']], palette='Purples_r')
plt.title('Chart 13: Department Demand Volatility (Coefficient of Variation: Std/Mean)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Coefficient of Variation (Demand Volatility)', fontsize=11)
plt.ylabel('Department', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart13_department_volatility.png')
plt.close()

# Chart 14: Correlation Heatmap
plt.figure(figsize=(12, 9))
corr_cols = ['Weekly_Sales', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
             'Total_MarkDown', 'IsHoliday', 'Is_Thanksgiving', 'Is_Christmas']
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, square=True)
plt.title('Chart 14: Multi-Variate Correlation Matrix of Macroeconomic & Store Drivers', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart14_correlation_heatmap.png')
plt.close()

# Chart 15: Pair Plot
pair_cols = ['Weekly_Sales', 'Size', 'CPI', 'Unemployment']
sample_pair = df[pair_cols].sample(n=2000, random_state=42)
g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 'color': '#2b5c8f'})
g.fig.suptitle('Chart 15: Pairwise Scatter Matrix of Core Variables', y=1.02, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/chart15_pair_plot.png')
plt.close()

print("All 15 EDA charts saved successfully!", flush=True)

# -------------------------------------------------------------
# 3. Hypothesis Testing
# -------------------------------------------------------------
print("\n[3/6] Running Formal Hypothesis Tests...", flush=True)

# Test 1: Holiday vs Non-Holiday Sales
holiday_sales = df[df['IsHoliday'] == 1]['Weekly_Sales']
non_holiday_sales = df[df['IsHoliday'] == 0]['Weekly_Sales']
t_stat, p_val_h = stats.ttest_ind(holiday_sales, non_holiday_sales, equal_var=False)
print(f"Hypothesis 1 (Two-Sample t-test): t={t_stat:.4f}, p-value={p_val_h:.4e}")

# Test 2: ANOVA across Store Types
type_A_sales = df[df['Type_A'] == 1]['Weekly_Sales']
type_B_sales = df[df['Type_B'] == 1]['Weekly_Sales']
type_C_sales = df[df['Type_C'] == 1]['Weekly_Sales']
f_stat, p_val_anova = stats.f_oneway(type_A_sales, type_B_sales, type_C_sales)
print(f"Hypothesis 2 (One-Way ANOVA): F={f_stat:.4f}, p-value={p_val_anova:.4e}")

# Test 3: Correlation between Unemployment and Sales
r_val, p_val_corr = stats.pearsonr(df['Unemployment'], df['Weekly_Sales'])
print(f"Hypothesis 3 (Pearson Correlation): r={r_val:.4f}, p-value={p_val_corr:.4e}")

# -------------------------------------------------------------
# 4. Anomaly Detection
# -------------------------------------------------------------
print("\n[4/6] Executing Anomaly Detection Models...", flush=True)
store_week = df.groupby(['Store', 'Date', 'IsHoliday']).agg({
    'Weekly_Sales': 'sum',
    'Total_MarkDown': 'first',
    'Temperature': 'first',
    'Fuel_Price': 'first',
    'CPI': 'first',
    'Unemployment': 'first',
    'Size': 'first'
}).reset_index()

stat_res = detect_statistical_anomalies(store_week)
store_week_iso, iso_model = detect_isolation_forest_anomalies(store_week)
store_week_cat = categorize_anomalies(store_week_iso)
joblib.dump(iso_model, 'models/isolation_forest_anomaly_model.joblib')

# Save Anomaly Chart
plt.figure(figsize=(14, 5))
chain_week = store_week_cat.groupby('Date').agg({
    'Weekly_Sales': 'sum',
    'Iso_Anomaly': 'sum'
}).reset_index()
plt.plot(chain_week['Date'], chain_week['Weekly_Sales'] / 1e6, color='#1f77b4', lw=2, label='Chain Sales ($M)')
anom_dates = chain_week[chain_week['Iso_Anomaly'] > 3]
plt.scatter(anom_dates['Date'], anom_dates['Weekly_Sales'] / 1e6, color='red', s=70, zorder=5, label='High-Anomaly Store Weeks (>3 stores)')
plt.title('Component 1: Multi-Dimensional Anomaly Detection (Isolation Forest vs Chain Sales)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Date', fontsize=11)
plt.ylabel('Weekly Sales ($ Millions)', fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig16_anomaly_detection.png')
plt.close()

# -------------------------------------------------------------
# 5. Store Segmentation & Market Basket Analysis
# -------------------------------------------------------------
print("\n[5/6] Executing Store Segmentation & Market Basket Analysis...", flush=True)

# Segmentation
store_profiles = build_store_profiles(df)
clust_df, scaler, kmeans_model, agg_model, clust_metrics = perform_clustering(store_profiles)
joblib.dump(kmeans_model, 'models/kmeans_segmentation_model.joblib')
joblib.dump(scaler, 'models/segmentation_scaler.joblib')

print(f"Store Segmentation Quality: Silhouette={clust_metrics['silhouette']:.4f}, Davies-Bouldin={clust_metrics['davies_bouldin']:.4f}, Calinski-Harabasz={clust_metrics['calinski_harabasz']:.2f}")

# Save Cluster Chart
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.scatterplot(data=clust_df, x='Size', y='Avg_Weekly_Sales', hue='Segment_Persona', palette='Set1', s=100, ax=axes[0])
axes[0].set_title('Store Clusters: Size vs Average Weekly Sales', fontweight='bold')
axes[0].set_xlabel('Store Physical Size (Sq. Ft.)')
axes[0].set_ylabel('Average Weekly Sales ($)')

# Dendrogram
Z = linkage(scaler.transform(store_profiles[[
    'Avg_Weekly_Sales', 'CV_Sales', 'Size', 'Sales_per_SqFt',
    'Avg_MarkDown', 'Mean_Unemployment', 'Mean_CPI', 'Return_Rate'
]]), method='ward')
dendrogram(Z, ax=axes[1], truncate_mode='lastp', p=12, leaf_rotation=45)
axes[1].set_title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontweight='bold')
axes[1].set_ylabel('Cluster Distance')
plt.suptitle('Component 2: Store Segmentation Profiling & Hierarchical Clustering', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig17_store_segmentation_clusters.png')
plt.close()

# Market Basket Analysis
basket_df, top_depts_list = construct_department_baskets(df, top_n_depts=20, surge_quantile=0.80)
itemsets, rules = mine_association_rules(basket_df, min_support=0.08, min_lift=1.2, max_len=3)
recs_df = extract_merchandising_recommendations(rules, top_k=10)
recs_df.to_csv('reports/market_basket_recommendations.csv', index=False)
print(f"Market Basket: {len(itemsets)} frequent itemsets, {len(rules)} association rules mined.")

# Save MBA Chart
plt.figure(figsize=(10, 5))
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', alpha=0.6, s=50)
plt.colorbar(label='Lift Ratio')
plt.title('Component 3: Association Rules Space (Support vs Confidence colored by Lift)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Rule Support', fontsize=11)
plt.ylabel('Rule Confidence', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig18_market_basket_rules.png')
plt.close()

# -------------------------------------------------------------
# 6. Demand Forecasting
# -------------------------------------------------------------
print("\n[6/6] Training & Evaluating Demand Forecasting Models...", flush=True)
train_df, test_df = split_train_test(df, split_date='2012-02-10')

feature_cols = [
    'Store', 'Dept', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Total_MarkDown', 'Has_MarkDown',
    'Year', 'Month', 'Week', 'Day',
    'IsHoliday', 'Is_Thanksgiving', 'Is_Christmas', 'Is_SuperBowl', 'Is_LaborDay',
    'Type_A', 'Type_B', 'Type_C',
    'Store_Dept_Avg', 'Store_Avg', 'Dept_Avg'
]

X_train = train_df[feature_cols].copy()
y_train = train_df['Weekly_Sales'].copy()
X_test = test_df[feature_cols].copy()
y_test = test_df['Weekly_Sales'].copy()

X_train['IsHoliday'] = X_train['IsHoliday'].astype(int)
X_test['IsHoliday'] = X_test['IsHoliday'].astype(int)

# Train all models
models, metrics, preds = train_forecasting_models(X_train, y_train, X_test, y_test)

# Baseline SARIMAX
sarimax_res, sarimax_preds, sarimax_metrics = fit_sarimax_chain_baseline(df)

# Print benchmark results
print("\n" + "="*50)
print("DEMAND FORECASTING BENCHMARK RESULTS")
print("="*50)
bench_df = pd.DataFrame(metrics).T
print(bench_df.to_string())
bench_df.to_csv('reports/forecasting_model_metrics.csv')

# Save Best Model
best_model_name = 'LightGBM' if metrics['LightGBM']['WMAE'] <= metrics['Random Forest']['WMAE'] else 'Random Forest'
joblib.dump(models[best_model_name], f'models/best_forecasting_model_{best_model_name.lower().replace(" ", "_")}.joblib')
joblib.dump(models, 'models/all_trained_models.joblib')
print(f"\nPersisted best model ({best_model_name}) to models/ directory.")

# Save Benchmark Comparison Chart
plt.figure(figsize=(12, 5))
metric_names = ['RMSE', 'MAE', 'WMAE']
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 5))
for i, m in enumerate(metric_names):
    vals = [metrics[model][m] for model in metrics]
    ax.bar(x + i*width, vals, width, label=m)

ax.set_title('Demand Forecasting Performance Benchmark (RMSE, MAE, WMAE)', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x + width)
ax.set_xticklabels(list(metrics.keys()), fontsize=11)
ax.set_ylabel('Error in Dollars ($)', fontsize=11)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig19_model_benchmarks_wmae.png')
plt.close()

# Save Feature Importances
plt.figure(figsize=(12, 6))
importances = models['LightGBM'].feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(15)
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='crest')
plt.title('Top 15 Feature Importances (LightGBM Gain Score)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Importance Score', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig20_feature_importances.png')
plt.close()

# Save Prediction vs Actual on Chain Aggregate
test_eval_df = test_df.copy()
test_eval_df['Pred_Sales'] = preds[best_model_name]

actual_chain = test_eval_df.groupby('Date')['Weekly_Sales'].sum() / 1e6
pred_chain = test_eval_df.groupby('Date')['Pred_Sales'].sum() / 1e6

plt.figure(figsize=(14, 5))
plt.plot(actual_chain.index, actual_chain.values, label='Actual Test Sales ($M)', color='navy', lw=2.2)
plt.plot(pred_chain.index, pred_chain.values, label=f'{best_model_name} Forecast ($M)', color='crimson', lw=2.2, linestyle='--')
plt.title(f'Component 4: Actual vs Out-of-Time Forecast (2012 Test Period) using {best_model_name}', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Date', fontsize=11)
plt.ylabel('Weekly Sales ($ Millions)', fontsize=11)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig21_predicted_vs_actual.png')
plt.close()

print("\n" + "="*60)
print("ALL ARTIFACTS, FIGURES, AND MODELS GENERATED SUCCESSFULLY!")
print("="*60, flush=True)
