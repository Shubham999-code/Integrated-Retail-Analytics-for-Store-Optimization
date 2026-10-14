# Integrated Retail Analytics for Store Optimization and Demand Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20LightGBM%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed%20%7C%20Evaluated-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end machine learning and business intelligence system designed to optimize retail store operations, analyze category affinities, detect demand anomalies, and forecast weekly sales across 45 stores and 81 departments.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Multi-Pillar Machine Learning Architecture](#multi-pillar-machine-learning-architecture)
3. [Dataset Architecture & Schema](#dataset-architecture--schema)
4. [Exploratory Data Analysis Highlights](#exploratory-data-analysis-highlights)
5. [Component 1: Multi-Dimensional Anomaly Detection](#component-1-multi-dimensional-anomaly-detection)
6. [Component 2: Store & Customer Segmentation](#component-2-store--customer-segmentation)
7. [Component 3: Market Basket Analysis & Association Rules](#component-3-market-basket-analysis--association-rules)
8. [Component 4: Demand Forecasting Benchmarks](#component-4-demand-forecasting-benchmarks)
9. [Component 5: Enterprise Retail Strategy](#component-5-enterprise-retail-strategy)
10. [Repository Directory Structure](#repository-directory-structure)
11. [Quickstart & Reproduction Guide](#quickstart--reproduction-guide)
12. [Deliverables Checklist](#deliverables-checklist)

---

## Project Overview

In physical retail management, profitability depends on balancing product availability, carrying costs, and promotional markdown efficiency. This project delivers an integrated analytical framework evaluating 143 historical trading weeks (421,570 records) of Walmart retail operations.

**Key Achievements:**
- **Champion Forecast Accuracy:** **Random Forest** achieved a Weighted Mean Absolute Error (**WMAE**) of **$1,816.13** ($R^2 = 0.9729$), reducing prediction error by **34.2%** over linear regularized baselines ($2,760.92).
- **Fast Production Inference:** **LightGBM** achieved near-identical accuracy (WMAE **$1,968.77**, $R^2 = 0.9709$) with **3.2s training time** (8x faster than Random Forest).
- **Domain Anomaly Treatment:** Isolated 194 multi-dimensional anomalies (3.01%) using Isolation Forest, attributing root causes to valid holiday rushes, markdown windfalls, weather shocks, and return waves.
- **Actionable Store Personas:** Segmented 45 locations into Flagship Supercenters, Suburban Growth Outlets, and Rural Value Hubs, quantitatively validated via Silhouette Score (0.222) and Davies-Bouldin Index (1.563).
- **High-Lift Category Merchandising:** Discovered 3,444 association rules via FP-Growth mining, surfacing grocery cross-shopping synergies with Lift > 5.0 and Confidence > 90%.

---

## Multi-Pillar Machine Learning Architecture

```
                                  RAW ENTERPRISE DATA
                  +--------------------------------------------------+
                  |  Stores (45)  |  Features (8.2k)  |  Sales (421k)|
                  +-------------------------+------------------------+
                                            |
                                            v
                                 PREPROCESSING & CLEANING
                  +--------------------------------------------------+
                  | - MarkDown Zero-Fill (Nov 2011 launch rationale) |
                  | - Return Flagging (Negative Sales Isolation)     |
                  | - Out-of-Time Chronological Split (Train < Feb12)|
                  +-------------------------+------------------------+
                                            |
         +--------------------+-------------+-------------+--------------------+
         |                    |                           |                    |
         v                    v                           v                    v
+------------------+ +------------------+  +-------------------+ +--------------------+
| ANOMALY DETECT.  | | STORE CLUSTERING |  | MARKET BASKET     | | DEMAND FORECAST    |
| - Statistical IQR| | - K-Means (k=3)  |  | - FP-Growth Mining| | - Ridge Baseline   |
| - IsolationForest| | - Agglomerative  |  | - 3,444 Rules     | | - Random Forest    |
| - Root Causes    | | - DBSCAN         |  | - Lift > 5.0      | | - LightGBM (Fast)  |
| - Return Tracking| | - 3 Store Personas| | - Endcap Bundles  | | - XGBoost          |
+--------+---------+ +--------+---------+  +---------+---------+ | - WMAE Optimization|
         |                    |                      |           +---------+----------+
         +--------------------+-------------+--------+---------------------+
                                            |
                                            v
                             EXECUTIVE RETAIL STRATEGY
                  +--------------------------------------------------+
                  | - Dynamic Markdown Schedules                     |
                  | - Volatility-Based Safety Stock Buffers          |
                  | - Layout Co-Merchandising & Endcap Placement     |
                  | - Macroeconomic Scenario & Drift Monitoring      |
                  +--------------------------------------------------+
```

---

## Dataset Architecture & Schema

The data integrates three historical enterprise files:
- **`stores data-set.csv` (45 rows)**: Store metadata (`Store`, `Type`: A, B, C, `Size` in sq. ft.).
- **`Features data set.csv` (8,190 rows)**: Environmental and economic drivers (`Date`, `Temperature`, `Fuel_Price`, `MarkDown1-5`, `CPI`, `Unemployment`, `IsHoliday`).
- **`sales data-set.csv` (421,570 rows)**: Historical departmental weekly sales (`Store`, `Dept`, `Date`, `Weekly_Sales`, `IsHoliday`).

---

## Exploratory Data Analysis Highlights

| Figure File | Focus Area | Key Business Insight |
| :--- | :--- | :--- |
| `reports/figures/chart01_weekly_sales_trend.png` | Weekly Sales Trend | Total volume spikes from $45M baseline to >$80M during Thanksgiving & Pre-Christmas. |
| `reports/figures/chart02_store_type_comparison.png` | Store Format Comparison | Type A Supercenters average $20,099/dept-wk vs Type C at $9,519/dept-wk ($F=7,764, p<10^{-15}$). |
| `reports/figures/chart03_top_departments.png` | Top 15 Departments | Depts 92, 95, 38, 72, 90, 40 drive Pareto concentration (> $250M sales each). |
| `reports/figures/chart05_holiday_distribution.png` | Holiday Sales Density | Statistically significant holiday sales premium ($t=7.00, p=2.59 \times 10^{-12}$, Welch's t-test). |
| `reports/figures/chart07_markdown_distribution.png` | Promotional MarkDowns | MarkDown1 ($1,080M) and MarkDown5 ($660M) receive over 70% of total promotional capital. |
| `reports/figures/chart09_unemployment_vs_sales.png` | Unemployment Elasticity | Negative correlation ($r=-0.0259, p<10^{-60}$); discretionary sales soften when unemployment > 10%. |
| `reports/figures/chart12_negative_sales_returns.png` | Reverse Logistics Returns | 1,285 negative sales records isolated, concentrated in consumer electronics post-holidays. |

---

## Component 1: Multi-Dimensional Anomaly Detection

- **Univariate Statistical Detection**: 1.5 $\times$ IQR rule flagged 34 store-weeks (0.53%).
- **Multivariate Isolation Forest**: Modeled 7 continuous operational drivers, identifying 194 anomalous store-weeks (3.01%).
- **Root-Cause Classification Framework**:
  - *Holiday Demand Surges (42 instances)*: Valid shopping peaks; weighted 5x in forecasting.
  - *Promotional Lifts (58 instances)*: Non-holiday spikes driven by markdown spend > 85th percentile.
  - *Operational Disruptions (24 instances)*: Foot traffic collapse due to blizzards (< 25°F) or stockouts.
  - *Reverse Logistics Return Spikes (52 instances)*: High net customer refunds in January.

---

## Component 2: Store & Customer Segmentation

Profiles were built using 8 standardized dimensions (Sales Volume, CV Volatility, Size, Sales/SqFt, Markdowns, Unemployment, CPI, Return Rate).

```
+---------------------------------------------------------------------------------------------------+
|                                      STORE CLUSTER PROFILES                                       |
+----------------------------------+----------------------------------+-----------------------------+
| Cluster 0: Flagship Supercenters | Cluster 1: Suburban Outlets      | Cluster 2: Rural Value Hubs |
| (18 Stores - 40% of Network)     | (16 Stores - 36% of Network)     | (11 Stores - 24% of Network)|
+----------------------------------+----------------------------------+-----------------------------+
| - Avg Sales: $20,845 / dept-wk   | - Avg Sales: $15,120 / dept-wk   | - Avg Sales: $9,650 / dept-wk|
| - Avg Size:  185,400 sq. ft.     | - Avg Size:  126,500 sq. ft.     | - Avg Size:  54,200 sq. ft. |
| - High Assortment Depth          | - Heavy Markdown Dependency      | - Essential Grocery Focus   |
| - Space Efficient ($0.11 / sqft) | - Price-Sensitive Shoppers       | - High Unemployment Impact  |
+----------------------------------+----------------------------------+-----------------------------+
```

**Quantitative Cluster Quality:**
- **Silhouette Score:** `0.2216`
- **Davies-Bouldin Index:** `1.5627`
- **Calinski-Harabasz Index:** `16.70`

---

## Component 3: Market Basket Analysis & Association Rules

Derived from $6,435$ weekly department baskets using **FP-Growth**:
- **708 Frequent Itemsets** mined in 0.11s.
- **3,444 Association Rules** discovered at Lift $\ge 1.2$.

**Top Association Rules:**
- `Dept [92, 94] (Produce + Bakery) -> Dept [93] (Frozen Foods)`: **Lift = 5.065**, **Confidence = 93.1%**, Support = 13.6%
- `Dept [8, 94] (Pets + Bakery) -> Dept [93] (Frozen Foods)`: **Lift = 5.051**, **Confidence = 92.9%**, Support = 8.1%
- `Dept [90, 38] (Dairy + Dry Grocery) -> Dept [93] (Frozen Foods)`: **Lift = 5.008**, **Confidence = 92.1%**, Support = 8.8%

---

## Component 4: Demand Forecasting Benchmarks

Evaluated on the out-of-time test partition (Feb 2012 – Oct 2012; 112,611 records):

| Algorithm | RMSE ($) | MAE ($) | WMAE ($) | $R^2$ Score | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SARIMAX Baseline** | $6,420.50 | $3,890.10 | $4,110.25 | 0.8812 | 1.5 s |
| **Ridge Regression** | $5,327.24 | $2,724.79 | $2,760.92 | 0.9421 | 0.6 s |
| **XGBoost Regressor** | $4,004.96 | $2,036.64 | $2,050.82 | 0.9673 | 2.2 s |
| **LightGBM Regressor** | $3,775.92 | $1,952.25 | $1,968.77 | 0.9709 | **3.2 s** |
| **Random Forest Regressor (Champion)** | **$3,643.14** | **$1,808.87** | **$1,816.13** | **0.9729** | 26.9 s |

$$\text{WMAE} = \frac{\sum w_i |y_i - \hat{y}_i|}{\sum w_i}, \quad w_i = 5 \text{ (Holiday)}, \quad w_i = 1 \text{ (Non-Holiday)}$$

---

## Component 5: Enterprise Retail Strategy

1. **Dynamic Markdown Timing**: Focus promotional budgets on Cluster 1 (Suburban Outlets) 2 weeks prior to holidays; protect staple margins in Cluster 0.
2. **Volatility-Adjusted Safety Stock**: Scale inventory buffers by 250% on high-volatility categories (Electronics, Toys; CV > 2.0) starting mid-October.
3. **Cross-Merchandising Layout**: Reconfigure endcaps to pair high-lift complementary categories (e.g. frozen dinner displays beside fresh produce).
4. **Drift Monitoring**: Monitor regional CPI and unemployment distributions to trigger automated model retraining when economic shocks occur.

---

## Repository Directory Structure

```
├── Integrated_Retail_Analytics_Store_Optimization.ipynb  # Fully executed Jupyter notebook with all outputs
├── PROJECT_REPORT.md                                      # Comprehensive 30+ page equivalent documentation
├── VIDEO_PRESENTATION_SCRIPT.md                           # Spoken script & interview Q&A guide
├── requirements.txt                                       # Pinned Python package dependencies
├── stores data-set.csv                                    # Stores raw metadata
├── Features data set.csv                                  # External economic features
├── sales data-set.csv                                     # Historical departmental weekly sales
├── models/                                                # Serialized production models
│   ├── best_forecasting_model.joblib                      # Champion Random Forest regressor
│   ├── lgbm_forecasting_model.joblib                      # Production LightGBM regressor
│   ├── isolation_forest_anomaly_model.joblib              # Unsupervised anomaly model
│   ├── kmeans_segmentation_model.joblib                   # Store clustering model
│   └── segmentation_scaler.joblib                         # Standard scaler pipeline
├── reports/                                               # Generated reports and figure outputs
│   ├── figures/                                           # 21 high-resolution visualization charts
│   ├── forecasting_model_metrics.csv                      # Tabular benchmark metrics
│   └── market_basket_recommendations.csv                  # Merchandising association rules
└── src/                                                   # Modular production source code
    ├── data_preprocessing.py                              # Ingestion, cleaning, target encoding
    ├── anomaly_detection.py                               # Statistical IQR & Isolation Forest
    ├── segmentation.py                                    # Store profiling, K-Means, DBSCAN
    ├── market_basket.py                                   # FP-Growth transaction basket mining
    └── forecasting.py                                     # Model training, WMAE evaluation
```

---

## Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
git clone https://github.com/shubham-kodape/Integrated-Retail-Analytics-Store-Optimization.git
cd Integrated-Retail-Analytics-Store-Optimization
pip install -r requirements.txt
```

### 2. Run the Full Analytics Pipeline
```bash
python src/generate_artifacts_and_figures.py
```

### 3. Open the Executed Jupyter Notebook
```bash
jupyter notebook Integrated_Retail_Analytics_Store_Optimization.ipynb
```

---

## Deliverables Checklist

- [x] **Individual Executed Jupyter Notebook** (`Integrated_Retail_Analytics_Store_Optimization.ipynb`): Strictly conforms to the AlmaBetter template; all 305+ cells executed with visible outputs, tables, and 24 embedded PNG plots.
- [x] **Unified GitHub Repository Structure** with clear `README.md`, `requirements.txt`, modular scripts in `src/`, models in `models/`, and charts in `reports/figures/`.
- [x] **Detailed Project Report** (`PROJECT_REPORT.md`): End-to-end documentation ready for Google Docs submission.
- [x] **Video Presentation Script & Interview Guide** (`VIDEO_PRESENTATION_SCRIPT.md`): Calibrated for 15–40 minute presentations with word-for-word talking points and answers to all 5 follow-up questions.

---
*Created by Shubham Kodape for the AlmaBetter Capstone Project.*
