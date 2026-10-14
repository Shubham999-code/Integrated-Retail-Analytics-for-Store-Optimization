"""
market_basket.py
----------------
Market Basket Analysis and Cross-Department Association Rule Mining
using FP-Growth and Apriori algorithms.
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

def construct_department_baskets(df, top_n_depts=20, surge_quantile=0.80):
    """
    Constructs weekly store-level department transaction baskets.
    A department is considered 'co-active/surging' in a given store-week
    if its weekly sales exceed the specified quantile for that department.
    """
    # Identify top N revenue-driving departments
    top_depts = df.groupby('Dept')['Weekly_Sales'].sum().nlargest(top_n_depts).index.tolist()
    df_top = df[df['Dept'].isin(top_depts)].copy()
    
    # Calculate surge threshold per department
    dept_thresholds = df_top.groupby('Dept')['Weekly_Sales'].transform(lambda x: x.quantile(surge_quantile))
    df_top['Surge'] = (df_top['Weekly_Sales'] > dept_thresholds).astype(int)
    
    # Pivot to binary basket matrix: Rows = (Store, Date), Columns = Dept
    basket_df = df_top.pivot_table(
        index=['Store', 'Date'],
        columns='Dept',
        values='Surge',
        fill_value=0
    ).astype(bool)
    
    return basket_df, top_depts

def mine_association_rules(basket_df, min_support=0.08, min_lift=1.2, max_len=3):
    """
    Mines frequent itemsets using FP-Growth and derives association rules.
    Computes Support, Confidence, Lift, Leverage, and Conviction.
    """
    frequent_itemsets = fpgrowth(basket_df, min_support=min_support, max_len=max_len, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
    
    # Format readable department strings
    rules['antecedent_str'] = rules['antecedents'].apply(lambda s: ', '.join([f"Dept {x}" for x in s]))
    rules['consequent_str'] = rules['consequents'].apply(lambda s: ', '.join([f"Dept {x}" for x in s]))
    
    return frequent_itemsets, rules

def extract_merchandising_recommendations(rules, top_k=10):
    """
    Extracts high-lift cross-selling recommendations from association rules.
    """
    top_rules = rules.head(top_k)
    recs = []
    for _, r in top_rules.iterrows():
        recs.append({
            'Trigger_Departments': r['antecedent_str'],
            'Associated_Departments': r['consequent_str'],
            'Support': round(r['support'], 3),
            'Confidence': round(r['confidence'], 3),
            'Lift': round(r['lift'], 3),
            'Strategic_Action': f"Place promotional endcaps and bundled coupons linking {r['antecedent_str']} with {r['consequent_str']}."
        })
    return pd.DataFrame(recs)
