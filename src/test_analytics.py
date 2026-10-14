import pandas as pd
import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from mlxtend.frequent_patterns import fpgrowth, association_rules
import warnings
warnings.filterwarnings('ignore')

print("--- Testing Anomaly Detection, Segmentation & Market Basket Analysis ---", flush=True)

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

# 1. Anomaly Detection Test
print("\n[1] Anomaly Detection:", flush=True)
store_week = df.groupby(['Store', 'Date', 'IsHoliday']).agg({
    'Weekly_Sales': 'sum',
    'Total_MarkDown': 'first',
    'Temperature': 'first',
    'Fuel_Price': 'first',
    'CPI': 'first',
    'Unemployment': 'first',
    'Size': 'first'
}).reset_index()

Q1 = store_week['Weekly_Sales'].quantile(0.25)
Q3 = store_week['Weekly_Sales'].quantile(0.75)
IQR = Q3 - Q1
stat_anomalies = (store_week['Weekly_Sales'] < (Q1 - 1.5 * IQR)) | (store_week['Weekly_Sales'] > (Q3 + 1.5 * IQR))
print(f"Statistical IQR Anomalies: {stat_anomalies.sum()} / {len(store_week)} ({stat_anomalies.mean()*100:.2f}%)", flush=True)

iso = IsolationForest(contamination=0.03, random_state=42)
iso_features = store_week[['Weekly_Sales', 'Total_MarkDown', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']]
store_week['Iso_Anomaly'] = iso.fit_predict(iso_features)
print(f"Isolation Forest Anomalies: {(store_week['Iso_Anomaly'] == -1).sum()} / {len(store_week)} ({(store_week['Iso_Anomaly'] == -1).mean()*100:.2f}%)", flush=True)
print(f"Negative sales transactions (Returns): {(sales['Weekly_Sales'] < 0).sum()}", flush=True)

# 2. Store Segmentation Test
print("\n[2] Store Segmentation:", flush=True)
store_profiles = df.groupby('Store').agg({
    'Weekly_Sales': ['mean', 'std', 'sum'],
    'Size': 'first',
    'Total_MarkDown': 'mean',
    'Unemployment': 'mean',
    'CPI': 'mean'
})
store_profiles.columns = ['Avg_Weekly_Sales', 'Sales_Std', 'Total_Sales', 'Size', 'Avg_MarkDown', 'Mean_Unemployment', 'Mean_CPI']
store_profiles['CV_Sales'] = store_profiles['Sales_Std'] / store_profiles['Avg_Weekly_Sales']
store_profiles['Sales_per_SqFt'] = store_profiles['Total_Sales'] / store_profiles['Size']

scaler = StandardScaler()
X_clust = scaler.fit_transform(store_profiles)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
store_profiles['KMeans_Cluster'] = kmeans.fit_predict(X_clust)

agg_clust = AgglomerativeClustering(n_clusters=3)
store_profiles['Agg_Cluster'] = agg_clust.fit_predict(X_clust)

dbscan = DBSCAN(eps=1.5, min_samples=3)
store_profiles['DBSCAN_Cluster'] = dbscan.fit_predict(X_clust)

sil_km = silhouette_score(X_clust, store_profiles['KMeans_Cluster'])
db_km = davies_bouldin_score(X_clust, store_profiles['KMeans_Cluster'])
ch_km = calinski_harabasz_score(X_clust, store_profiles['KMeans_Cluster'])

print(f"K-Means (k=3): Silhouette={sil_km:.4f}, Davies-Bouldin={db_km:.4f}, Calinski-Harabasz={ch_km:.2f}", flush=True)
print("Cluster distribution:\n", store_profiles['KMeans_Cluster'].value_counts(), flush=True)

# 3. Market Basket Analysis Test
print("\n[3] Market Basket Analysis (Department Level Associations):", flush=True)
# Co-surge definition: Top 20 departments that have a weekly surge (> 80th percentile for that dept)
top_depts = df.groupby('Dept')['Weekly_Sales'].sum().nlargest(20).index.tolist()
df_top = df[df['Dept'].isin(top_depts)].copy()

dept_p80 = df_top.groupby('Dept')['Weekly_Sales'].transform(lambda x: x.quantile(0.80))
df_top['Surge'] = (df_top['Weekly_Sales'] > dept_p80).astype(int)

basket = df_top.pivot_table(index=['Store', 'Date'], columns='Dept', values='Surge', fill_value=0).astype(bool)
print(f"Basket matrix shape: {basket.shape}", flush=True)

t0 = time.time()
frequent_itemsets = fpgrowth(basket, min_support=0.08, max_len=3, use_colnames=True)
print(f"Frequent itemsets found ({time.time()-t0:.2f}s): {len(frequent_itemsets)}", flush=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules = rules.sort_values('lift', ascending=False)
print(f"Association rules mined: {len(rules)}", flush=True)
print("Top 5 association rules:", flush=True)
for _, r in rules.head(5).iterrows():
    ants = list(r['antecedents'])
    cons = list(r['consequents'])
    print(f"  Dept {ants} -> Dept {cons} | Supp: {r['support']:.3f}, Conf: {r['confidence']:.3f}, Lift: {r['lift']:.3f}", flush=True)

print("\n--- All tests completed successfully! ---", flush=True)
