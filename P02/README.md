# DemandForest — New Product Demand Forecasting Pipeline

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E.svg)
![Jupyter](https://img.shields.io/badge/notebook-Jupyter-F37626.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research%20prototype-yellow.svg)

An end-to-end implementation of **DemandForest**, a pre-launch demand forecasting method for new products with no historical sales, adapted and extended for a real-world B2B distribution dataset (dental consumables and equipment).

> **Reference methodology:** van Steenbergen, R.M. & Mes, M.R.K. (2020). *Forecasting demand profiles of new products.* Decision Support Systems, 139, 113401. https://doi.org/10.1016/j.dss.2020.113401

## Overview

Forecasting demand for a brand-new product is hard: there's no sales history to extrapolate from. DemandForest solves this by learning from products that *have* already launched. It splits the forecasting problem into two parts:

1. **Shape** — what does the demand curve look like over the introduction period (front-loaded, stable, back-loaded)?
2. **Scale** — how much will be sold in total?

These are predicted separately from pre-launch product characteristics (category, brand, price, supplier, etc.) using clustering and tree-based ensembles, then combined into a full weekly forecast with prediction intervals for inventory planning.

This notebook reimplements the original paper's pipeline and extends it with:

- **Google Trends** as a pre-launch demand signal (brand-level search interest)
- **Gamma and Log-Normal distribution fitting** over the empirical Quantile Regression Forest output, to smooth prediction intervals when few comparable products exist
- **Chronological (time-aware) train/validation/test splitting** to simulate a realistic cold-start setting, instead of a random split
- **Winkler Score** as an additional interval-quality metric alongside PICP/PINAW
- **Permutation feature importance** for both the profile classifier and the demand regressor
- Outlier trimming on the training set to control upper-bound inflation
- An expanded hyperparameter search grid for the Random Forest classifier

## Methodology

| Step | Technique | Purpose |
|---|---|---|
| 1. Profile extraction | Normalize weekly sales of existing products | Derive the shape `p_x` of each product's demand curve, independent of scale |
| 2. Clustering | K-Means (k chosen via Davies-Bouldin, Silhouette, Calinski-Harabasz) | Group products into a small number of representative demand profiles |
| 3. Profile classification | Random Forest classifier | Predict which profile cluster a *new* product will follow, from its launch-time features |
| 4. Total demand regression | Quantile Regression Forest | Predict the new product's total demand over the horizon, plus full conditional quantiles |
| 5. Distribution fitting | Gamma / Log-Normal MLE fit on QRF quantiles | Smooth the empirical prediction interval, especially with sparse comparable products |
| 6. Synthesis | Predicted profile × predicted total demand | Produce the final per-week forecast and prediction interval |

Forecasts and intervals are evaluated with RMSE, WMAPE, PICP (coverage), PINAW (interval width), Winkler Score, plus classification accuracy and Cohen's Kappa for the profile assignment.

## Notebook Structure

| Section | Contents |
|---|---|
| 1. Environment & Path Configuration | Global parameters: horizon length, tree counts, train/val/test split shares |
| 2. Profile Extraction & Data Preparation | `extract_demand_profiles()` — builds normalized weekly demand profiles and total demand per product |
| 3. K-Means Clustering & Optimal K Selection | `find_optimal_k()`, `cluster_profiles()` — parallel sweep over k with majority-vote/weighted-average selection |
| 4. Model Training: RF & QRF | `build_preprocessor()`, `train_demand_forest()` — grid-searched Random Forest classifier + Quantile Regression Forest regressor |
| 5. Forecasting & Interval Synthesis | `generate_forecasts()`, `fit_theoretical_bounds()` — combines predicted profile and total demand; fits Gamma/Log-Normal bounds |
| 6. Evaluation Metrics | `evaluate_forecast()`, `evaluate_intervals()`, `winkler_score()`, `wmape()` |
| 7. Execution Pipeline Wrapper | `extract_split_features()`, `run_demandforest_pipeline()` — orchestrates one full train→predict→evaluate cycle, including Google Trends merge |
| 8. Temporal Splitting | `_split_timestamps()`, `prepare_demandforest_data()` — chronological cutoffs for cold-start validation/test sets |
| 9. Execution Wrapper | `wrapper_demandforest()` — runs validation then test phases, consolidates metrics, exports Excel results |
| 10. Global Parameters & Execution | Loads the dataset, defines pre-launch features, triggers the full pipeline |

## Data

The pipeline expects a transaction-level dataset (one row per product per sales event) with at minimum:

- A product identifier (`product_code_base`)
- A timestamp (`invoice_date`) and a derived first-sale / launch date (`first_sale`)
- A quantity column (`item_quantity`)
- Pre-launch product characteristics — in this notebook: `brand_name`, `Producer`, `Region`, `Category`, `full_sub_sub_cat`, `full_sub_cat`, `is_new_brand`, `num_similars`, `substitutability`, `price_index`, `holiday`, `pre_launch_trend_score`

**Important:** only features known *before* launch are used as predictors (`PRE_LAUNCH_FEATURES`). Post-launch behavioral signals are deliberately excluded, since they wouldn't be available at the time a real forecast is needed.

The notebook downloads its working dataset (`dataset.pkl`) from a Google Drive link and caches Google Trends data from a Kaggle dataset input path if available, falling back to live API calls otherwise.

A categories reference table (`categories_only.xlsx`) is included for mapping the category / sub-category / sub-sub-category hierarchy used in this domain (dental equipment, consumables, etc.), including local-language category names.

## Requirements

```
pandas
numpy
scipy
scikit-learn
quantile-forest
joblib
matplotlib
seaborn
requests
tqdm
pytrends
```

Install with:

```bash
pip install pandas numpy scipy scikit-learn quantile-forest joblib matplotlib seaborn requests tqdm pytrends
```

(The notebook also installs `quantile-forest` and `pytrends` inline via `!pip install`.)

## Usage

Run all cells top to bottom. The final cell:

1. Downloads/loads the dataset
2. Defines pre-launch feature columns and columns to drop
3. Calls `wrapper_demandforest(...)`, which:
   - Cleans and prepares the data
   - Splits chronologically into train / validation / test by product launch date
   - Runs the full DemandForest pipeline on the validation split, then again on train+validation → test
   - Prints a consolidated metrics table (Validation vs. Test)
   - Exports three timestamped Excel files: validation results, test results, and metrics summary

Key tunables at the top of the notebook:

```python
test_share = 0.2            # fraction of products held out for testing (by launch date)
val_share_of_train = 0.2    # fraction of the remaining products held out for validation
FORECAST_HORIZON_WEEKS = 12 # length of the introduction period to forecast
N_TREES = 2000               # trees in the Random Forest / Quantile Regression Forest
RANDOM_STATE = 42
k_min, k_max = 2, 10         # range of cluster counts evaluated for K-Means
```

## Outputs

Running the pipeline with `flag_export=True` produces:

- `DemandForest_Validation_Results_<timestamp>.xlsx` — per-product point forecast and 90% interval (validation phase)
- `DemandForest_Test_Results_<timestamp>.xlsx` — same, for the held-out test phase
- `DemandForest_Metrics_<timestamp>.xlsx` — side-by-side validation vs. test metrics (RMSE, WMAPE, Kappa, accuracy, PICP/PINAW/Winkler for both empirical and Gamma-fitted intervals, tuned hyperparameters, and feature importances)

## Known Deviations from the Original Paper

- **K selection:** uses a weighted/normalized average of the three cluster validity indices when they disagree, rather than a simple majority vote, since a plain vote tended to collapse to `k_min`.
- **Interval quality metric:** adds the Winkler Score, which penalizes both excessive interval width and actual non-coverage, as a complement to PINAW.
- **Splitting strategy:** uses a strict chronological split on product launch date rather than a random 75/25 split, to better simulate genuine cold-start forecasting.
- **Extra feature:** incorporates a Google Trends–based `pre_launch_trend_score` not present in the original paper.

## Citation

If you use or build on this pipeline, please cite the original methodology paper:

```bibtex
@article{vansteenbergen2020demandforest,
  title   = {Forecasting demand profiles of new products},
  author  = {van Steenbergen, R.M. and Mes, M.R.K.},
  journal = {Decision Support Systems},
  volume  = {139},
  pages   = {113401},
  year    = {2020},
  doi     = {10.1016/j.dss.2020.113401}
}
```
