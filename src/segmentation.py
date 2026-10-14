"""
segmentation.py
---------------
Store and Department Segmentation analysis using K-Means,
Agglomerative Hierarchical Clustering, and DBSCAN.
Evaluates cluster homogeneity and separation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def build_store_profiles(df):
    """
    Constructs multi-dimensional store profiles for clustering:
    - Avg_Weekly_Sales: Revenue scale
    - Sales_Std: Sales dispersion
    - CV_Sales: Coefficient of Variation (volatility/seasonality)
    - Size: Store physical size in square feet
    - Sales_per_SqFt: Efficiency / productivity
    - Avg_MarkDown: Promotional dependency
    - Mean_Unemployment: Macro sensitivity
    - Mean_CPI: Local price environment
    - Return_Rate: Proportion of negative sales transactions
    """
    profiles = df.groupby('Store').agg({
        'Weekly_Sales': ['mean', 'std', 'sum'],
        'Size': 'first',
        'Total_MarkDown': 'mean',
        'Unemployment': 'mean',
        'CPI': 'mean',
        'Is_Return': 'mean'
    })
    profiles.columns = [
        'Avg_Weekly_Sales', 'Sales_Std', 'Total_Sales',
        'Size', 'Avg_MarkDown', 'Mean_Unemployment', 'Mean_CPI', 'Return_Rate'
    ]
    profiles['CV_Sales'] = profiles['Sales_Std'] / (profiles['Avg_Weekly_Sales'] + 1e-5)
    profiles['Sales_per_SqFt'] = profiles['Total_Sales'] / (profiles['Size'] + 1e-5)
    
    return profiles

def perform_clustering(store_profiles, n_clusters=3, random_state=42):
    """
    Fits K-Means, Agglomerative Clustering, and DBSCAN on standardized store profiles.
    Returns clustered DataFrame, fitted models, and validation metrics.
    """
    features = [
        'Avg_Weekly_Sales', 'CV_Sales', 'Size', 'Sales_per_SqFt',
        'Avg_MarkDown', 'Mean_Unemployment', 'Mean_CPI', 'Return_Rate'
    ]
    X = store_profiles[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=15)
    km_labels = kmeans.fit_predict(X_scaled)
    
    # 2. Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    agg_labels = agg.fit_predict(X_scaled)
    
    # 3. DBSCAN
    dbscan = DBSCAN(eps=2.0, min_samples=3)
    db_labels = dbscan.fit_predict(X_scaled)
    
    # Metrics for K-Means
    metrics = {
        'silhouette': silhouette_score(X_scaled, km_labels),
        'davies_bouldin': davies_bouldin_score(X_scaled, km_labels),
        'calinski_harabasz': calinski_harabasz_score(X_scaled, km_labels)
    }
    
    res_df = store_profiles.copy()
    res_df['KMeans_Segment'] = km_labels
    res_df['Agg_Segment'] = agg_labels
    res_df['DBSCAN_Segment'] = db_labels
    
    # Assign intuitive persona names based on average weekly sales and size
    cluster_means = res_df.groupby('KMeans_Segment')['Avg_Weekly_Sales'].mean()
    sorted_clusters = cluster_means.sort_values(ascending=False).index.tolist()
    
    persona_mapping = {
        sorted_clusters[0]: 'Cluster 0: Flagship Supercenters (High Volume, High Size)',
        sorted_clusters[1]: 'Cluster 1: Suburban Growth Outlets (Moderate Volume, Promotional)',
        sorted_clusters[2]: 'Cluster 2: Rural Value Hubs (Low Volume, Recession Sensitive)'
    }
    res_df['Segment_Persona'] = res_df['KMeans_Segment'].map(persona_mapping)
    
    return res_df, scaler, kmeans, agg, metrics
