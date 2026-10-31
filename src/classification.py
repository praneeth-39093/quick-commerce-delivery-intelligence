"""
Classification Module for Late-Order Detection.
Identifies orders at risk of breaching the SLA delivery threshold.
Implements multi-model pipelines (Logistic Regression, Random Forest, XGBoost),
class imbalance handling, threshold optimization, and probability scoring.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_recall_curve

import config
from src.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LateOrderClassifier:
    """Classifies order breach risks to trigger early operational interventions."""

    SUPPORTED_ALGORITHMS = ["logistic_regression", "random_forest", "xgboost"]

    def __init__(self, algorithm: str = "xgboost"):
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose from {self.SUPPORTED_ALGORITHMS}")
        self.algorithm = algorithm
        self.pipeline: Optional[Pipeline] = None
        self.feature_engineer = FeatureEngineer()
        self.is_fitted: bool = False
        self.optimal_threshold: float = 0.50

    def _get_base_estimator(self, scale_pos_weight: float = 1.0, **kwargs) -> Any:
        """Instantiates classification estimator with imbalance adjustments."""
        random_state = config.MODEL_CONFIG.random_state

        if self.algorithm == "logistic_regression":
            c_param = kwargs.get("C", 1.0)
            return LogisticRegression(
                C=c_param,
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state
            )

        elif self.algorithm == "random_forest":
            n_estimators = kwargs.get("n_estimators", 150)
            max_depth = kwargs.get("max_depth", 12)
            n_jobs = kwargs.get("n_jobs", config.MODEL_CONFIG.n_jobs)
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight="balanced",
                n_jobs=n_jobs,
                random_state=random_state
            )

        elif self.algorithm == "xgboost":
            n_estimators = kwargs.get("n_estimators", 200)
            max_depth = kwargs.get("max_depth", 6)
            learning_rate = kwargs.get("learning_rate", 0.05)
            subsample = kwargs.get("subsample", 0.8)
            n_jobs = kwargs.get("n_jobs", config.MODEL_CONFIG.n_jobs)
            return XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                scale_pos_weight=scale_pos_weight,
                n_jobs=n_jobs,
                random_state=random_state,
                eval_metric="logloss"
            )

    def build_pipeline(self, scale_pos_weight: float = 1.0, **model_kwargs) -> Pipeline:
        """Assembles preprocessing and classifier into unified Pipeline."""
        preprocessor = self.feature_engineer.build_preprocessing_pipeline()
        estimator = self._get_base_estimator(scale_pos_weight=scale_pos_weight, **model_kwargs)

        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator)
        ])
        return self.pipeline

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **model_kwargs) -> Pipeline:
        """
        Fits classification pipeline with class-imbalance ratio adjustment.

        Args:
            X_train: Training features.
            y_train: Binary indicator (1 = Late delivery, 0 = On-time).

        Returns:
            Fitted Pipeline.
        """
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        imbalance_ratio = float(neg_count / max(pos_count, 1))

        logger.info(
            f"Training Late-Order Classifier ({self.algorithm}). "
            f"Class ratio (Negative/Positive): {imbalance_ratio:.2f}"
        )

        self.build_pipeline(scale_pos_weight=imbalance_ratio, **model_kwargs)
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("Late-Order Classifier training completed.")
        return self.pipeline

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Generates binary class predictions with threshold tuning."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call train() or load_model() first.")

        decision_threshold = threshold if threshold is not None else self.optimal_threshold
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= decision_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns late-delivery breach probability distribution."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call train() or load_model() first.")
        return self.pipeline.predict_proba(X)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = config.MODEL_CONFIG.cv_folds,
        scoring: str = "roc_auc"
    ) -> Dict[str, float]:
        """Performs stratified K-fold cross-validation."""
        logger.info(f"Executing Stratified {cv}-fold cross-validation ({scoring})...")
        pipeline = self.build_pipeline()
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.MODEL_CONFIG.random_state)
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=config.MODEL_CONFIG.n_jobs)

        return {
            "cv_mean_score": float(np.mean(scores)),
            "cv_std_score": float(np.std(scores)),
            "cv_scores": [float(s) for s in scores]
        }

    def optimize_decision_threshold(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """Calculates decision threshold that maximizes F1-score on validation data."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)

        if best_idx < len(thresholds):
            self.optimal_threshold = float(thresholds[best_idx])
            logger.info(f"Optimized decision threshold identified: {self.optimal_threshold:.3f}")
        return self.optimal_threshold

    def save_model(self, filepath: Optional[Path] = None) -> Path:
        """Serializes trained classification pipeline."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Cannot save unfitted model.")

        save_path = Path(filepath) if filepath else config.MODELS_DIR / f"late_classifier_{self.algorithm}.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "threshold": self.optimal_threshold}, save_path)
        logger.info(f"Saved trained classification model to: {save_path}")
        return save_path

    def load_model(self, filepath: Path) -> Pipeline:
        """Loads serialized classification pipeline and tuned threshold."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at: {filepath}")

        loaded = joblib.load(filepath)
        if isinstance(loaded, dict) and "pipeline" in loaded:
            self.pipeline = loaded["pipeline"]
            self.optimal_threshold = loaded.get("threshold", 0.50)
        else:
            self.pipeline = loaded
            self.optimal_threshold = 0.50

        self.is_fitted = True
        logger.info(f"Loaded classification model from: {filepath}")
        return self.pipeline

