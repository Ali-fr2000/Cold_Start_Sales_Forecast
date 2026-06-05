# P02: K-Medoids + KNN Pipeline for Cold-Start Sales Forecasting

**Version:** `P02_KMEDOIDS_KNN`

## Overview

This pipeline implements an alternative clustering-based machine learning approach to forecast total sales of new products during their launch phase (first 8 weeks). Building upon the P01 baseline, P02 replaces the K-Means + Logistic Regression + Cluster-Mean architecture with:

- **K-Medoids (PAM)** for more robust clustering
- **K-Nearest Neighbors (KNN)** as the classifier
- **KNN Regression** as the direct sales forecaster

## Key Differences from P01

| Component | P01 | P02 |
|-----------|-----|-----|
| **Clustering** | K-Means | K-Medoids (PAM) |
| **Classifier** | Logistic Regression | KNN Classifier |
| **Forecaster** | Cluster-Mean Baseline | KNN Regression |
| **Robustness** | Sensitive to outliers | Robust to outliers |
| **Hyperparameters** | k (clusters) | k (clusters) + n_neighbors (KNN) |
| **Forecast Type** | Indirect (cluster-based) | Direct (neighbor-based) |

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Loading & Prep                      │
├─────────────────────────────────────────────────────────────┤
│  • Temporal filtering (first 8 weeks)                       │
│  • Feature engineering (price index, substitutability)      │
│  • Inflation adjustment                                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│         Temporal Splits (Train / Val / Test)                │
├─────────────────────────────────────────────────────────────┤
│  • Chronological split by product launch date               │
│  • Train: 64%, Validation: 16%, Test: 20%                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│         K-Medoids Clustering (Training Products)            │
├─────────────────────────────────────────────────────────────┤
│  • Uses PAM (Partitioning Around Medoids) algorithm         │
│  • More robust than K-Means to outliers                     │
│  • Selects actual data points as cluster centers            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────┐
│  KNN Classifier: Predict Cluster (New Products)              │
├─────────────────────────────────────────────────────────────┤
│  • Learns cluster membership from training set               │
│  • Uses launch-time features only                            │
│  • Distance-weighted KNN voting                              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│     KNN Regression: Direct Sales Forecast (8-week total)     │
├─────────────────────────────────────────────────────────────┤
│  • Predicts 8-week sales directly from neighbors             │
│  • Weighted by inverse distance                              │
│  • No intermediate cluster assignment needed                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Evaluation Metrics (WMAPE, RMSE, MAE)           │
├─────────────────────────────────────────────────────────────┤
│  • Weighted Mean Absolute Percentage Error                   │
│  • Root Mean Square Error                                    │
│  • Mean Absolute Error                                       │
│  • Bias analysis                                             │
└─────────────────────────────────────────────────────────────┘
```

## Features Used

### Clustering Features (K-Medoids)
All launch-time and post-launch behavioral features:
- `price_index` - Inflation-adjusted unit price
- `Region` - Geographic region
- `Category`, `full_sub_cat` - Product categorization
- `num_similars`, `substitutability` - Competitive intensity
- `is_new_brand` - Brand launch status
- `launch_month` - Seasonal indicator
- `duration_weeks`, `duration_days` - Tenure metrics
- `total_sales_quantity`, `num_sales_transactions` - Sales activity
- `median_weekly_quantity`, `cv_quantity` - Sales volatility
- `median_discount_rate` - Pricing strategy

### Classification Features (KNN Classifier)
Launch-time features only (excludes post-launch behavior):
- `price_index`
- `Region`
- `Category`, `full_sub_cat`
- `num_similars`, `substitutability`
- `is_new_brand`
- `launch_month`

### Regression Features (KNN Regressor)
Launch-time features for direct 8-week sales prediction:
- `price_index`
- `Region`
- `Category`, `full_sub_cat`
- `num_similars`, `substitutability`
- `is_new_brand`
- `launch_month`

## Hyperparameter Tuning

The notebook performs a comprehensive grid search to find optimal parameters:

```python
k_min = 2
k_max = 100                              # Number of clusters to test
n_neighbors_list = [3, 5, 7, 9, 11]    # Number of neighbors for KNN
```

**Optimization Criteria:**
1. Primary: Minimize WMAPE (Weighted Mean Absolute Percentage Error)
2. Secondary: For similar WMAPEs (±0.01), prefer smaller `k` and `n_neighbors` (Occam's Razor)

## Execution Workflow

### Section 10: Validation Phase
```python
# Finds optimal (k, n_neighbors) using validation set
all_detailed_val_results, sorted_results, opt_k, opt_n_neighbors, opt_detailed_val_results = wrapper(...)
```
- Trains on 64% historical products
- Validates on 16% new products (intermediate launch dates)
- Returns top performing configurations

### Section 11: Test Phase
```python
# Re-trains with optimal parameters using train+val combined
# Evaluates on 20% test products (most recent launches)
all_detailed_test_results, sorted_test_results, opt_k, opt_n_neighbors, opt_detailed_test_results = wrapper(...)
```

## Key Advantages of K-Medoids + KNN

| Advantage | Benefit |
|-----------|---------|
| **K-Medoids Robustness** | Handles outlier products better than K-Means |
| **Actual Medoids** | Cluster centers are real products, more interpretable |
| **KNN Flexibility** | No assumption of cluster shapes; adapts to local density |
| **Direct Forecasting** | KNN regression captures non-linear relationships |
| **Distance Weighting** | Closer neighbors have stronger influence |
| **No Cluster Assignment** | Avoids cascading errors from cluster mismatch |

## Known Limitations

- **Cold-Start on Cold-Start:** Still requires historical training data for new categories
- **Feature Selection:** Launch-time features may not capture all product characteristics
- **Curse of Dimensionality:** Performance may degrade with very high-dimensional feature spaces
- **Computational Cost:** K-Medoids is O(n²) per iteration (slower than K-Means)
- **KNN Scaling:** All distance calculations at prediction time (slower for large datasets)

## Future Improvements

1. **Adaptive K Selection:** Use silhouette score or elbow method instead of grid search
2. **Distance Metrics:** Test cosine, mahalanobis, or learned distances
3. **Weighted Features:** Use feature importance from random forests to weight KNN
4. **Ensemble Methods:** Combine P01 and P02 predictions for robustness
5. **Category-Specific Models:** Separate pipelines per product category
6. **Temporal Features:** Include seasonality, trend, and cyclical components
7. **Imbalance Handling:** Address skewed distributions in new product types

## File Structure

```
P02/
├── GH_kmedoids_knn_documented.ipynb     # Main pipeline notebook
├── README.md                             # This file
└── (Output files generated at runtime)
    ├── P02_kmedoids_knn_results_*.xlsx
    └── P02_kmedoids_knn_results_detailed_*.xlsx
```

## Dependencies

```python
# Core
pandas, numpy

# ML
scikit-learn, sklearn-extra (for KMedoids)
scipy

# Utilities
pathlib, datetime, pytz, joblib, multiprocessing

# Viz (optional)
matplotlib, seaborn
```

**Install sklearn-extra:**
```bash
pip install scikit-learn-extra
```

## Running the Pipeline

### Google Colab (Recommended)
1. Upload notebook to Colab
2. Mount Google Drive
3. Update `folder_path` and `out_folder` paths
4. Run all cells sequentially

### Local Execution
```python
# Ensure all dependencies installed
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook P02/GH_kmedoids_knn_documented.ipynb
```

## Output Interpretation

### Macro Results (`P02_kmedoids_knn_results_*.xlsx`)
Columns:
- `k` - Number of clusters
- `n_neighbors` - Number of neighbors for KNN
- `wmape` - Weighted Mean Absolute Percentage Error
- `rmse` - Root Mean Square Error
- `mae` - Mean Absolute Error
- `bias` - Systematic over/under forecast

**Lower WMAPE = Better Forecast Accuracy**

### Detailed Results (`P02_kmedoids_knn_results_detailed_*.xlsx`)
Columns:
- `product_code` - Product identifier
- `total_8w_qty` - Actual 8-week sales quantity
- `predicted_8w_qty` - KNN regression forecast

## Comparison with P01

Run both pipelines on the same dataset and compare:

```python
import pandas as pd

p01_results = pd.read_excel('P01_results.xlsx')
p02_results = pd.read_excel('P02_results.xlsx')

print(f"P01 Best WMAPE: {p01_results['wmape'].min():.4f}")
print(f"P02 Best WMAPE: {p02_results['wmape'].min():.4f}")
print(f"Improvement: {(p01_results['wmape'].min() - p02_results['wmape'].min()) / p01_results['wmape'].min() * 100:.2f}%")
```

## Citation

If you use this pipeline in research, please cite:

```bibtex
@thesis{ghassemi2026coldstart,
  title={Cold-Start Sales Forecasting Using K-Medoids and KNN},
  author={Ghassemi, Mirali},
  school={Politecnico di Milano},
  year={2026},
  note={MSc Thesis - Supply Chain Analytics}
}
```

## Contact

For questions or issues with the P02 pipeline, please open an issue in the repository or contact the maintainers.

---

**Last Updated:** 2026-06-05  
**Pipeline Version:** P02_KMEDOIDS_KNN  
**Status:** Active Development
