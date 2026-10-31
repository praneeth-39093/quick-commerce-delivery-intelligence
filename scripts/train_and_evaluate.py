"""
End-to-End Model Training, Hyperparameter Tuning, Evaluation & Artifact Generation Script.
Executes Chronological Splitting, ETA Regression, Late-Order Classification,
Dark-Store Spatial Clustering, Rider Behavioral Segmentation, SHAP Explainability,
and Artifact Serialization.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Redirect matplotlib config to temp directory to silence sandbox warnings
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, silhouette_score
)
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shap

import config
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model_evaluation import ModelEvaluator
from src.clustering import DeliveryOptimizerClustering

# Configure visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.sans-serif": "Arial", "font.family": "sans-serif"})

print("=" * 80)
print("  PHASE 3: MODEL TRAINING, HYPERPARAMETER TUNING & BENCHMARK COMPARISON")
print("=" * 80)

# ==============================================================================
# STEP 1: CHRONOLOGICAL DATA INGESTION & LEAK-FREE SPLITTING
# ==============================================================================
print("\n[1/6] Ingesting featured dataset and performing chronological train/val/test split...")
loader = DataLoader()
featured_path = config.PROCESSED_DATA_PATH

if not featured_path.exists():
    raise FileNotFoundError(f"Featured dataset not found at: {featured_path}")

df_featured = pd.read_csv(featured_path)
print(f"Loaded featured dataset: {df_featured.shape[0]} rows, {df_featured.shape[1]} columns.")

train_df, val_df, test_df = loader.split_chronological(df_featured)
print(f"  • Training set:   {len(train_df):5d} rows (Order dates < {config.MODEL_CONFIG.train_end_date})")
print(f"  • Validation set: {len(val_df):5d} rows (Order dates {config.MODEL_CONFIG.train_end_date} to {config.MODEL_CONFIG.val_end_date})")
print(f"  • Test set:       {len(test_df):5d} rows (Order dates >= {config.MODEL_CONFIG.val_end_date})")

# Feature matrix extraction using FeatureEngineer
engineer = FeatureEngineer(sla_threshold_min=config.DEFAULT_SLA_MINUTES)
X_train, y_train_reg = engineer.get_feature_matrices(train_df, task="regression")
_, y_train_clf = engineer.get_feature_matrices(train_df, task="classification")

X_val, y_val_reg = engineer.get_feature_matrices(val_df, task="regression")
_, y_val_clf = engineer.get_feature_matrices(val_df, task="classification")

X_test, y_test_reg = engineer.get_feature_matrices(test_df, task="regression")
_, y_test_clf = engineer.get_feature_matrices(test_df, task="classification")

print(f"Feature set ($X$): {list(X_train.columns)}")
print(f"Number of order-placement features: {X_train.shape[1]} (Strictly leakage-free)")

# Build and fit preprocessing pipeline strictly on training split
preprocessor = engineer.build_preprocessing_pipeline()
preprocessor.fit(X_train)

X_train_proc = preprocessor.transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)
print(f"Transformed feature matrix dimension: {X_train_proc.shape[1]} encoded features.")

evaluator = ModelEvaluator()
comparison_records = []

# ==============================================================================
# STEP 2: PART 1 - ETA REGRESSION MODELS
# ==============================================================================
print("\n[2/6] Training & Tuning ETA Regression Models...")

# 1. Mean Baseline Regressor
print("\n--- Training Model 1: Mean Baseline Regressor ---")
t0 = time.time()
base_reg = DummyRegressor(strategy="mean")
base_reg.fit(X_train_proc, y_train_reg)
base_train_time = time.time() - t0

y_val_pred_base = base_reg.predict(X_val_proc)
y_test_pred_base = base_reg.predict(X_test_proc)

base_metrics_val = evaluator.evaluate_regression(y_val_reg, y_val_pred_base, "Mean Baseline Regressor (Val)", "mean_baseline")
base_metrics_test = evaluator.evaluate_regression(y_test_reg, y_test_pred_base, "Mean Baseline Regressor (Test)", "mean_baseline")
comparison_records.append({
    **base_metrics_test,
    "split": "test",
    "val_mae": base_metrics_val["mae"],
    "val_rmse": base_metrics_val["rmse"],
    "val_r2": base_metrics_val["r2_score"],
    "training_time_sec": round(base_train_time, 2)
})

# 2. Ridge Regression with Validation Tuning
print("\n--- Training Model 2: Ridge Regression (Tuning alpha on Val) ---")
best_alpha = 1.0
best_ridge_val_rmse = float("inf")

for alpha in [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 200.0]:
    ridge_trial = Ridge(alpha=alpha, random_state=config.MODEL_CONFIG.random_state)
    ridge_trial.fit(X_train_proc, y_train_reg)
    trial_pred = ridge_trial.predict(X_val_proc)
    trial_rmse = np.sqrt(mean_squared_error(y_val_reg, trial_pred))
    if trial_rmse < best_ridge_val_rmse:
        best_ridge_val_rmse = trial_rmse
        best_alpha = alpha

print(f"Optimal Ridge alpha selected: {best_alpha} (Val RMSE: {best_ridge_val_rmse:.3f}m)")
t0 = time.time()
best_ridge = Ridge(alpha=best_alpha, random_state=config.MODEL_CONFIG.random_state)
best_ridge.fit(X_train_proc, y_train_reg)
ridge_train_time = time.time() - t0

y_val_pred_ridge = best_ridge.predict(X_val_proc)
y_test_pred_ridge = best_ridge.predict(X_test_proc)

ridge_metrics_val = evaluator.evaluate_regression(y_val_reg, y_val_pred_ridge, "Ridge Regression (Val)", "ridge")
ridge_metrics_test = evaluator.evaluate_regression(y_test_reg, y_test_pred_ridge, "Ridge Regression (Test)", "ridge")
comparison_records.append({
    **ridge_metrics_test,
    "split": "test",
    "val_mae": ridge_metrics_val["mae"],
    "val_rmse": ridge_metrics_val["rmse"],
    "val_r2": ridge_metrics_val["r2_score"],
    "training_time_sec": round(ridge_train_time, 2),
    "best_params": f"alpha={best_alpha}"
})

# 3. Random Forest Regressor with Validation Tuning
print("\n--- Training Model 3: Random Forest Regressor (Tuning Depth & Estimators) ---")
rf_param_grid = [
    {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 4},
    {"n_estimators": 150, "max_depth": 15, "min_samples_leaf": 4},
    {"n_estimators": 150, "max_depth": 20, "min_samples_leaf": 2}
]
best_rf_params = None
best_rf_val_rmse = float("inf")

for params in rf_param_grid:
    rf_trial = RandomForestRegressor(
        **params,
        random_state=config.MODEL_CONFIG.random_state,
        n_jobs=-1
    )
    rf_trial.fit(X_train_proc, y_train_reg)
    val_pred = rf_trial.predict(X_val_proc)
    val_rmse = np.sqrt(mean_squared_error(y_val_reg, val_pred))
    print(f"  RF trial {params} -> Val RMSE: {val_rmse:.3f}m")
    if val_rmse < best_rf_val_rmse:
        best_rf_val_rmse = val_rmse
        best_rf_params = params

print(f"Optimal RF parameters selected: {best_rf_params}")
t0 = time.time()
best_rf = RandomForestRegressor(
    **best_rf_params,
    random_state=config.MODEL_CONFIG.random_state,
    n_jobs=-1
)
best_rf.fit(X_train_proc, y_train_reg)
rf_train_time = time.time() - t0

y_val_pred_rf = best_rf.predict(X_val_proc)
y_test_pred_rf = best_rf.predict(X_test_proc)

rf_metrics_val = evaluator.evaluate_regression(y_val_reg, y_val_pred_rf, "Random Forest Regressor (Val)", "random_forest")
rf_metrics_test = evaluator.evaluate_regression(y_test_reg, y_test_pred_rf, "Random Forest Regressor (Test)", "random_forest")
comparison_records.append({
    **rf_metrics_test,
    "split": "test",
    "val_mae": rf_metrics_val["mae"],
    "val_rmse": rf_metrics_val["rmse"],
    "val_r2": rf_metrics_val["r2_score"],
    "training_time_sec": round(rf_train_time, 2),
    "best_params": str(best_rf_params)
})

# 4. XGBoost Regressor with Validation Tuning
print("\n--- Training Model 4: XGBoost Regressor (Tuning Learning Rate & Depth) ---")
xgb_param_grid = [
    {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 250, "max_depth": 7, "learning_rate": 0.03, "subsample": 0.85}
]
best_xgb_params = None
best_xgb_val_rmse = float("inf")

for params in xgb_param_grid:
    xgb_trial = XGBRegressor(
        **params,
        random_state=config.MODEL_CONFIG.random_state,
        n_jobs=-1,
        tree_method="hist"
    )
    xgb_trial.fit(X_train_proc, y_train_reg)
    val_pred = xgb_trial.predict(X_val_proc)
    val_rmse = np.sqrt(mean_squared_error(y_val_reg, val_pred))
    print(f"  XGB trial {params} -> Val RMSE: {val_rmse:.3f}m")
    if val_rmse < best_xgb_val_rmse:
        best_xgb_val_rmse = val_rmse
        best_xgb_params = params

print(f"Optimal XGBoost parameters selected: {best_xgb_params}")
t0 = time.time()
best_xgb_reg = XGBRegressor(
    **best_xgb_params,
    random_state=config.MODEL_CONFIG.random_state,
    n_jobs=-1,
    tree_method="hist"
)
best_xgb_reg.fit(X_train_proc, y_train_reg)
xgb_train_time = time.time() - t0

y_val_pred_xgb = best_xgb_reg.predict(X_val_proc)
y_test_pred_xgb = best_xgb_reg.predict(X_test_proc)

xgb_reg_metrics_val = evaluator.evaluate_regression(y_val_reg, y_val_pred_xgb, "XGBoost Regressor (Val)", "xgboost")
xgb_reg_metrics_test = evaluator.evaluate_regression(y_test_reg, y_test_pred_xgb, "XGBoost Regressor (Test)", "xgboost")
comparison_records.append({
    **xgb_reg_metrics_test,
    "split": "test",
    "val_mae": xgb_reg_metrics_val["mae"],
    "val_rmse": xgb_reg_metrics_val["rmse"],
    "val_r2": xgb_reg_metrics_val["r2_score"],
    "training_time_sec": round(xgb_train_time, 2),
    "best_params": str(best_xgb_params)
})

# ==============================================================================
# STEP 3: PART 2 - LATE-ORDER CLASSIFICATION MODELS
# ==============================================================================
print("\n[3/6] Training & Tuning Late-Order Classification Models...")

neg_count = np.sum(y_train_clf == 0)
pos_count = np.sum(y_train_clf == 1)
imbalance_ratio = float(neg_count / max(pos_count, 1))
print(f"Training Class distribution: Negatives={neg_count}, Positives={pos_count} (Ratio: {imbalance_ratio:.2f})")

# Helper to optimize decision threshold on validation set
def optimize_threshold(y_true, y_probs):
    thresholds = np.linspace(0.15, 0.85, 71)
    best_f1 = -1.0
    best_thresh = 0.50
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
    return float(round(best_thresh, 3)), float(round(best_f1, 4))

# 1. Logistic Regression
print("\n--- Training Model 1: Logistic Regression ---")
t0 = time.time()
log_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.MODEL_CONFIG.random_state)
log_clf.fit(X_train_proc, y_train_clf)
log_train_time = time.time() - t0

y_val_probs_log = log_clf.predict_proba(X_val_proc)[:, 1]
opt_thresh_log, opt_f1_log = optimize_threshold(y_val_clf, y_val_probs_log)
print(f"Logistic Regression: Optimal Val Threshold={opt_thresh_log} (Val F1={opt_f1_log:.4f})")

y_test_probs_log = log_clf.predict_proba(X_test_proc)[:, 1]
y_test_pred_log = (y_test_probs_log >= opt_thresh_log).astype(int)

log_metrics_test = evaluator.evaluate_classification(
    y_test_clf, y_test_pred_log, y_test_probs_log,
    model_name="Logistic Regression Classifier", algorithm="logistic_regression"
)
comparison_records.append({
    **log_metrics_test,
    "split": "test",
    "optimal_threshold": opt_thresh_log,
    "val_f1": opt_f1_log,
    "training_time_sec": round(log_train_time, 2)
})

# 2. Random Forest Classifier
print("\n--- Training Model 2: Random Forest Classifier ---")
t0 = time.time()
rf_clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=14,
    class_weight="balanced",
    random_state=config.MODEL_CONFIG.random_state,
    n_jobs=-1
)
rf_clf.fit(X_train_proc, y_train_clf)
rf_clf_train_time = time.time() - t0

y_val_probs_rf = rf_clf.predict_proba(X_val_proc)[:, 1]
opt_thresh_rf, opt_f1_rf = optimize_threshold(y_val_clf, y_val_probs_rf)
print(f"Random Forest Classifier: Optimal Val Threshold={opt_thresh_rf} (Val F1={opt_f1_rf:.4f})")

y_test_probs_rf = rf_clf.predict_proba(X_test_proc)[:, 1]
y_test_pred_rf = (y_test_probs_rf >= opt_thresh_rf).astype(int)

rf_metrics_test = evaluator.evaluate_classification(
    y_test_clf, y_test_pred_rf, y_test_probs_rf,
    model_name="Random Forest Classifier", algorithm="random_forest"
)
comparison_records.append({
    **rf_metrics_test,
    "split": "test",
    "optimal_threshold": opt_thresh_rf,
    "val_f1": opt_f1_rf,
    "training_time_sec": round(rf_clf_train_time, 2)
})

# 3. XGBoost Classifier
print("\n--- Training Model 3: XGBoost Classifier ---")
t0 = time.time()
xgb_clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    scale_pos_weight=imbalance_ratio,
    random_state=config.MODEL_CONFIG.random_state,
    n_jobs=-1,
    eval_metric="logloss"
)
xgb_clf.fit(X_train_proc, y_train_clf)
xgb_clf_train_time = time.time() - t0

y_val_probs_xgb = xgb_clf.predict_proba(X_val_proc)[:, 1]
opt_thresh_xgb, opt_f1_xgb = optimize_threshold(y_val_clf, y_val_probs_xgb)
print(f"XGBoost Classifier: Optimal Val Threshold={opt_thresh_xgb} (Val F1={opt_f1_xgb:.4f})")

y_test_probs_xgb = xgb_clf.predict_proba(X_test_proc)[:, 1]
y_test_pred_xgb = (y_test_probs_xgb >= opt_thresh_xgb).astype(int)

xgb_metrics_test = evaluator.evaluate_classification(
    y_test_clf, y_test_pred_xgb, y_test_probs_xgb,
    model_name="XGBoost Classifier", algorithm="xgboost"
)
comparison_records.append({
    **xgb_metrics_test,
    "split": "test",
    "optimal_threshold": opt_thresh_xgb,
    "val_f1": opt_f1_xgb,
    "training_time_sec": round(xgb_clf_train_time, 2)
})

# Save updated benchmark comparisons
df_comparison = pd.DataFrame(comparison_records)
df_comparison.to_csv(config.METRICS_PATH, index=False)
print(f"\nSaved updated benchmark results to: {config.METRICS_PATH}")

# ==============================================================================
# STEP 4: PART 3 - DARK-STORE ZONE CLUSTERING
# ==============================================================================
print("\n[4/6] Executing Dark-Store Demand Zone Spatial Clustering...")
cluster_engine = DeliveryOptimizerClustering()

# Filter valid coordinates strictly
lat_min, lat_max = config.LATITUDE_BOUNDS
lon_min, lon_max = config.LONGITUDE_BOUNDS
valid_mask = (
    df_featured["delivery_location_latitude"].between(lat_min, lat_max) &
    df_featured["delivery_location_longitude"].between(lon_min, lon_max)
)
valid_coords_df = df_featured.loc[valid_mask].copy()

# Compute elbow and silhouette metrics
elbow_metrics = cluster_engine.evaluate_elbow_and_silhouette(valid_coords_df, k_range=range(2, 9))

# Plot Elbow and Silhouette curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(elbow_metrics["k_values"], elbow_metrics["inertias"], marker="o", color="#1f77b4", linewidth=2)
axes[0].set_title("Elbow Curve: Inertia vs Cluster Count (K)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (Sum of Squared Distances)")

axes[1].plot(elbow_metrics["k_values"], elbow_metrics["silhouette_scores"], marker="s", color="#2ca02c", linewidth=2)
axes[1].set_title("Silhouette Scores across K Values", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Average Silhouette Coefficient")

plt.tight_layout()
elbow_fig_path = config.FIGURES_DIR / "dark_store_elbow_silhouette.png"
plt.savefig(elbow_fig_path, dpi=300)
plt.close()
print(f"Saved Elbow/Silhouette plot to: {elbow_fig_path}")

# Fit optimal spatial clusters (K = 5 exploratory micro-zones)
optimal_k_zones = 5
clustered_spatial_df = cluster_engine.fit_spatial_clusters(df_featured, n_clusters=optimal_k_zones)
centroids_df = cluster_engine.get_cluster_centroids()

# Save spatial cluster scatter figure
plt.figure(figsize=(10, 8))
valid_clustered = clustered_spatial_df[clustered_spatial_df["delivery_zone_cluster"] >= 0]
sample_plot = valid_clustered.sample(min(len(valid_clustered), 4000), random_state=42)

scatter = plt.scatter(
    sample_plot["delivery_location_longitude"],
    sample_plot["delivery_location_latitude"],
    c=sample_plot["delivery_zone_cluster"],
    cmap="tab10", alpha=0.4, s=15, edgecolors="none"
)
# Overlay Centroids
plt.scatter(
    centroids_df["hub_longitude"],
    centroids_df["hub_latitude"],
    c="black", marker="*", s=220, edgecolors="white", linewidth=1.5,
    label="Exploratory Demand Centroid"
)
plt.title("Spatial Delivery Demand Zones & Candidate Centroids\n(Exploratory Micro-Fulfillment Zones)", fontsize=13, fontweight="bold")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc="upper right")
plt.tight_layout()
spatial_fig_path = config.FIGURES_DIR / "dark_store_clusters.png"
plt.savefig(spatial_fig_path, dpi=300)
plt.close()
print(f"Saved Spatial Cluster Map to: {spatial_fig_path}")

# ==============================================================================
# STEP 5: PART 4 - RIDER BEHAVIORAL SEGMENTATION
# ==============================================================================
print("\n[5/6] Executing Delivery Partner Behavioral Segmentation...")
rider_df, segment_profile_df = cluster_engine.profile_delivery_partners(df_featured, n_clusters=3)

# Plot Rider Segments
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=rider_df,
    x="avg_delivery_time",
    y="avg_rating",
    hue="efficiency_tier_name",
    palette=["#2ca02c", "#1f77b4", "#d62728"],
    alpha=0.75, s=60
)
plt.title("Delivery Partner Segmentation: Speed vs Rating Performance Tiers", fontsize=13, fontweight="bold")
plt.xlabel("Average Delivery Duration (minutes)")
plt.ylabel("Average Customer Rating (1-5 Scale)")
plt.legend(title="Partner Performance Tier", loc="lower left")
plt.tight_layout()
rider_fig_path = config.FIGURES_DIR / "rider_segments.png"
plt.savefig(rider_fig_path, dpi=300)
plt.close()
print(f"Saved Rider Segmentation plot to: {rider_fig_path}")

# ==============================================================================
# STEP 6: PART 5 & 6 - MODEL EXPLAINABILITY (SHAP & IMPORTANCES)
# ==============================================================================
print("\n[6/6] Computing Feature Importances & SHAP Explanations...")

# Get feature names from preprocessor
cat_encoder = preprocessor.named_transformers_["cat"]
encoded_cat_names = list(cat_encoder.get_feature_names_out(config.MODEL_CONFIG.categorical_features))
all_feature_names = config.MODEL_CONFIG.numerical_features + encoded_cat_names

# XGBoost Regression Feature Importances
reg_importances = pd.Series(best_xgb_reg.feature_importances_, index=all_feature_names).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
reg_importances.head(12).plot(kind="barh", color="#1f77b4").invert_yaxis()
plt.title("Top 12 Predictive Features for Delivery ETA (XGBoost)", fontsize=13, fontweight="bold")
plt.xlabel("Gini / Split Feature Importance")
plt.tight_layout()
reg_imp_path = config.FIGURES_DIR / "feature_importance_regression.png"
plt.savefig(reg_imp_path, dpi=300)
plt.close()

# SHAP Explanations for XGBoost Regressor
print("Computing SHAP values on validation sample...")
sample_idx = np.random.choice(len(X_val_proc), min(len(X_val_proc), 500), replace=False)
X_val_sample = X_val_proc[sample_idx]

explainer = shap.TreeExplainer(best_xgb_reg)
shap_values_reg = explainer(X_val_sample)
shap_values_reg.feature_names = all_feature_names

plt.figure(figsize=(11, 7))
shap.plots.beeswarm(shap_values_reg, max_display=12, show=False)
plt.title("SHAP Feature Attributions: Impact on Delivery Duration (ETA)", fontsize=13, fontweight="bold")
plt.tight_layout()
shap_reg_path = config.FIGURES_DIR / "shap_summary_regression.png"
plt.savefig(shap_reg_path, dpi=300)
plt.close()
print(f"Saved Regression SHAP Beeswarm to: {shap_reg_path}")

# SHAP Explanations for XGBoost Classifier
explainer_clf = shap.TreeExplainer(xgb_clf)
shap_values_clf = explainer_clf(X_val_sample)
shap_values_clf.feature_names = all_feature_names

plt.figure(figsize=(11, 7))
shap.plots.beeswarm(shap_values_clf, max_display=12, show=False)
plt.title("SHAP Feature Attributions: Impact on Late-Order Risk Probability", fontsize=13, fontweight="bold")
plt.tight_layout()
shap_clf_path = config.FIGURES_DIR / "shap_summary_classification.png"
plt.savefig(shap_clf_path, dpi=300)
plt.close()
print(f"Saved Classification SHAP Beeswarm to: {shap_clf_path}")

# ==============================================================================
# STEP 7: SERIALIZE TRAINED ARTIFACTS IN models/
# ==============================================================================
print("\nSaving finalized pipelines and model artifacts to models/...")
# Best regression pipeline
best_reg_pipeline = {
    "preprocessor": preprocessor,
    "model": best_xgb_reg,
    "algorithm": "xgboost",
    "feature_names": all_feature_names,
    "test_mae": xgb_metrics_test["mae"],
    "test_rmse": xgb_metrics_test["rmse"],
    "test_r2": xgb_metrics_test["r2_score"]
}
reg_model_file = config.MODELS_DIR / "best_eta_regressor.joblib"
joblib.dump(best_reg_pipeline, reg_model_file)
print(f"  • Saved Best Regressor to: {reg_model_file}")

# Best classification pipeline
best_clf_pipeline = {
    "preprocessor": preprocessor,
    "model": xgb_clf,
    "algorithm": "xgboost",
    "threshold": opt_thresh_xgb,
    "feature_names": all_feature_names,
    "test_precision": xgb_metrics_test["precision"],
    "test_recall": xgb_metrics_test["recall"],
    "test_f1": xgb_metrics_test["f1_score"],
    "test_roc_auc": xgb_metrics_test["roc_auc"],
    "test_pr_auc": xgb_metrics_test["pr_auc"]
}
clf_model_file = config.MODELS_DIR / "best_late_classifier.joblib"
joblib.dump(best_clf_pipeline, clf_model_file)
print(f"  • Saved Best Classifier to: {clf_model_file}")

# Clustering models
darkstore_cluster_artifact = {
    "kmeans_model": cluster_engine.kmeans_model,
    "centroids": centroids_df,
    "n_clusters": optimal_k_zones
}
ds_file = config.MODELS_DIR / "dark_store_kmeans.joblib"
joblib.dump(darkstore_cluster_artifact, ds_file)
print(f"  • Saved Dark Store Cluster model to: {ds_file}")

rider_cluster_artifact = {
    "kmeans_model": cluster_engine.kmeans_model,
    "segment_profiles": segment_profile_df,
    "scaler": cluster_engine.scaler
}
rider_file = config.MODELS_DIR / "rider_segmentation_kmeans.joblib"
joblib.dump(rider_cluster_artifact, rider_file)
print(f"  • Saved Rider Segmentation model to: {rider_file}")

# Roundtrip model loading validation
print("\nValidating model loading roundtrip...")
loaded_reg = joblib.load(reg_model_file)
loaded_clf = joblib.load(clf_model_file)
loaded_ds = joblib.load(ds_file)
loaded_rider = joblib.load(rider_file)
assert loaded_reg["model"] is not None
assert loaded_clf["model"] is not None
assert loaded_ds["centroids"] is not None
assert loaded_rider["segment_profiles"] is not None
print("✅ Verified model loading roundtrip: All artifacts deserialize correctly.")

print("\n" + "=" * 80)
print("TRAINING PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 80)
