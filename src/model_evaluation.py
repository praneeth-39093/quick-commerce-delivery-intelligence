"""
Model Evaluation and Performance Benchmarking Module.
Computes comprehensive statistical metrics for regression and classification tasks,
manages benchmark logging to reports/model_comparison.csv, and visualizes error diagnostics.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve
)

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelEvaluator:
    """Standardized metrics computation and comparative ledger management."""

    def __init__(self, comparison_path: Optional[Path] = None):
        self.comparison_path = Path(comparison_path) if comparison_path else config.METRICS_PATH

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "ETARegressor",
        algorithm: str = "xgboost"
    ) -> Dict[str, Any]:
        """
        Computes regression error metrics for delivery duration prediction.

        Args:
            y_true: Ground truth delivery duration in minutes.
            y_pred: Model predicted delivery duration in minutes.
            model_name: Identifier for model.
            algorithm: Algorithm variant.

        Returns:
            Dict containing MAE, RMSE, R2, and MAPE metrics.
        """
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)

        metrics = {
            "model_name": model_name,
            "task_type": "regression",
            "algorithm": algorithm,
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2_score": round(r2, 4),
            "mape_percent": round(mape, 2),
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Regression Evaluation [{algorithm}]: MAE={mae:.2f}m, RMSE={rmse:.2f}m, R²={r2:.4f}")
        return metrics

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: Optional[np.ndarray] = None,
        model_name: str = "LateOrderClassifier",
        algorithm: str = "xgboost"
    ) -> Dict[str, Any]:
        """
        Computes classification metrics for late-delivery risk detection.

        Args:
            y_true: Ground truth binary breach indicators.
            y_pred: Predicted class labels (0 or 1).
            y_probs: Predicted breach probability scores.
            model_name: Model identifier.
            algorithm: Algorithm variant.

        Returns:
            Dict containing Accuracy, Precision, Recall, F1, and ROC-AUC.
        """
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        roc_auc = float(roc_auc_score(y_true, y_probs)) if y_probs is not None else np.nan
        pr_auc = float(average_precision_score(y_true, y_probs)) if y_probs is not None else np.nan
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        metrics = {
            "model_name": model_name,
            "task_type": "classification",
            "algorithm": algorithm,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
            "pr_auc": round(pr_auc, 4) if not np.isnan(pr_auc) else None,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(
            f"Classification Evaluation [{algorithm}]: "
            f"Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}"
        )
        return metrics

    def append_to_comparison_table(self, metrics: Dict[str, Any]) -> None:
        """Logs evaluated metrics into model_comparison.csv."""
        df_row = pd.DataFrame([metrics])

        if self.comparison_path.exists() and self.comparison_path.stat().st_size > 0:
            existing = pd.read_csv(self.comparison_path)
            updated = pd.concat([existing, df_row], ignore_index=True)
        else:
            updated = df_row

        self.comparison_path.parent.mkdir(parents=True, exist_ok=True)
        updated.to_csv(self.comparison_path, index=False)
        logger.info(f"Recorded evaluation metrics in: {self.comparison_path}")

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str = "regression_residuals.png"
    ) -> Path:
        """Visualizes prediction errors and residual distributions."""
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter of predictions vs residuals
        axes[0].scatter(y_pred, residuals, alpha=0.3, color="#1f77b4", edgecolor="none")
        axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
        axes[0].set_title("Predicted Value vs Residuals", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Predicted Delivery Time (min)")
        axes[0].set_ylabel("Residual (Actual - Predicted)")

        # Distribution of residuals
        sns.histplot(residuals, kde=True, ax=axes[1], color="#2ca02c", bins=30)
        axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
        axes[1].set_title("Residual Distribution (Error Spread)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Residual (min)")

        plt.tight_layout()
        save_path = config.FIGURES_DIR / filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str = "classification_confusion_matrix.png"
    ) -> Path:
        """Plots annotated confusion matrix for late order predictions."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["On-Time (0)", "Late (1)"],
            yticklabels=["On-Time (0)", "Late (1)"]
        )
        plt.title("Late-Delivery Confusion Matrix", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.tight_layout()

        save_path = config.FIGURES_DIR / filename
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

