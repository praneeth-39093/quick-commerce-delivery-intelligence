"""
Script to generate evaluation curves: Actual vs Predicted, Residuals, ROC & PR curves,
and Permutation Importance.
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from sklearn.inspection import permutation_importance

import config
from src.data_loader import DataLoader

def main():
    print("Generating detailed evaluation curves...")
    df = pd.read_csv(config.DATA_DIR / "featured_delivery_data.csv")
    from src.feature_engineering import FeatureEngineer
    loader = DataLoader()
    engineer = FeatureEngineer()

    _, _, test_df = loader.split_chronological(
        df, "order_date", config.MODEL_CONFIG.train_end_date, config.MODEL_CONFIG.val_end_date
    )

    X_test, y_test_reg = engineer.get_feature_matrices(test_df, task="regression")
    _, y_test_clf = engineer.get_feature_matrices(test_df, task="classification")

    # Load models
    reg_bundle = joblib.load(config.MODELS_DIR / "best_eta_regressor.joblib")
    clf_bundle = joblib.load(config.MODELS_DIR / "best_late_classifier.joblib")

    preprocessor = reg_bundle["preprocessor"]
    X_test_proc = preprocessor.transform(X_test)

    # 1. Regression: Actual vs Predicted & Residuals
    y_test_pred_reg = reg_bundle["model"].predict(X_test_proc)
    residuals = y_test_reg - y_test_pred_reg

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    axes[0].scatter(y_test_reg, y_test_pred_reg, alpha=0.25, color="#1f77b4", edgecolors="none", s=20)
    min_val, max_val = min(y_test_reg.min(), y_test_pred_reg.min()), max(y_test_reg.max(), y_test_pred_reg.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Identity (Perfect Fit)")
    axes[0].set_title(f"XGBoost Regressor: Actual vs. Predicted ETA (Test Set)\nMAE={reg_bundle['test_mae']:.2f}m | RMSE={reg_bundle['test_rmse']:.2f}m | R²={reg_bundle['test_r2']:.4f}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Actual Delivery Time (min)", fontsize=10)
    axes[0].set_ylabel("Predicted Delivery Time (min)", fontsize=10)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Residual Distribution
    axes[1].hist(residuals, bins=50, color="#2ca02c", edgecolor="black", alpha=0.7, density=True)
    axes[1].axvline(0, color="red", linestyle="--", lw=2, label="Zero Error")
    axes[1].set_title("Residual Error Distribution (Actual - Predicted)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Residual (minutes)", fontsize=10)
    axes[1].set_ylabel("Density", fontsize=10)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    reg_curve_path = config.FIGURES_DIR / "regression_actual_vs_predicted.png"
    plt.savefig(reg_curve_path, dpi=300)
    plt.close()
    print(f"Saved: {reg_curve_path}")

    # 2. Classification: ROC and PR Curves
    clf_model = clf_bundle["model"]
    y_test_probs_clf = clf_model.predict_proba(X_test_proc)[:, 1]

    fpr, tpr, _ = roc_curve(y_test_clf, y_test_probs_clf)
    roc_auc_val = auc(fpr, tpr)

    precision_vals, recall_vals, _ = precision_recall_curve(y_test_clf, y_test_probs_clf)
    pr_auc_val = average_precision_score(y_test_clf, y_test_probs_clf)
    base_rate = np.mean(y_test_clf)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    axes[0].plot(fpr, tpr, color="#d62728", lw=2, label=f"XGBoost Classifier (AUC = {roc_auc_val:.4f})")
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Chance (AUC = 0.5000)")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=10)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=10)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Precision-Recall Curve
    axes[1].plot(recall_vals, precision_vals, color="#9467bd", lw=2, label=f"XGBoost Classifier (PR-AUC = {pr_auc_val:.4f})")
    axes[1].axhline(base_rate, color="k", linestyle="--", lw=1.5, label=f"No-Skill Baseline ({base_rate*100:.1f}%)")
    axes[1].set_title("Precision-Recall (PR) Curve", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    clf_curve_path = config.FIGURES_DIR / "classification_roc_pr_curves.png"
    plt.savefig(clf_curve_path, dpi=300)
    plt.close()
    print(f"Saved: {clf_curve_path}")

    # 3. Permutation Importance for Best Regressor
    print("Computing permutation importance for Best Regressor on test set...")
    perm_result = permutation_importance(
        reg_bundle["model"], X_test_proc, y_test_reg,
        n_repeats=5, random_state=42, n_jobs=-1, scoring="neg_mean_absolute_error"
    )
    feature_names = reg_bundle["feature_names"]
    perm_sorted_idx = perm_result.importances_mean.argsort()[::-1][:15]

    plt.figure(figsize=(10, 6))
    top_features = [feature_names[i] for i in perm_sorted_idx]
    top_importances = perm_result.importances_mean[perm_sorted_idx]
    top_stds = perm_result.importances_std[perm_sorted_idx]

    plt.barh(range(len(top_features)), top_importances[::-1], xerr=top_stds[::-1], align="center", color="#3b528b", ecolor="#5ec962", capsize=3)
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.title("Permutation Feature Importance (Test Set - MAE Impact)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean MAE Degradation When Shuffled (minutes)", fontsize=10)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    perm_path = config.FIGURES_DIR / "permutation_importance_regression.png"
    plt.savefig(perm_path, dpi=300)
    plt.close()
    print(f"Saved: {perm_path}")
    print("All additional evaluation curves generated successfully.")

if __name__ == "__main__":
    main()
