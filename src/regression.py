"""
Regression Module for ETA (Estimated Time of Arrival) Prediction.
Implements multi-model training pipelines (Ridge, Random Forest, XGBoost),
cross-validation, hyperparameter tuning, model persistence, and SHAP explainability.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold

import config
from src.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ETARegressor:
    """End-to-end regression trainer and predictor for Quick-Commerce Delivery Duration."""

    SUPPORTED_ALGORITHMS = ["mean_baseline", "ridge", "random_forest", "xgboost"]

    def __init__(self, algorithm: str = "xgboost"):
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose from {self.SUPPORTED_ALGORITHMS}")
        self.algorithm = algorithm
        self.pipeline: Optional[Pipeline] = None
        self.feature_engineer = FeatureEngineer()
        self.is_fitted: bool = False

    def _get_base_estimator(self, **kwargs) -> Any:
        """Instantiates selected regression estimator."""
        random_state = config.MODEL_CONFIG.random_state

        if self.algorithm == "mean_baseline":
            return DummyRegressor(strategy="mean")

        elif self.algorithm == "ridge":
            alpha = kwargs.get("alpha", 1.0)
            return Ridge(alpha=alpha, random_state=random_state)

        elif self.algorithm == "random_forest":
            n_estimators = kwargs.get("n_estimators", 150)
            max_depth = kwargs.get("max_depth", 12)
            n_jobs = kwargs.get("n_jobs", config.MODEL_CONFIG.n_jobs)
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=n_jobs,
                random_state=random_state
            )

        elif self.algorithm == "xgboost":
            n_estimators = kwargs.get("n_estimators", 200)
            max_depth = kwargs.get("max_depth", 6)
            learning_rate = kwargs.get("learning_rate", 0.05)
            subsample = kwargs.get("subsample", 0.8)
            n_jobs = kwargs.get("n_jobs", config.MODEL_CONFIG.n_jobs)
            return XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                n_jobs=n_jobs,
                random_state=random_state,
                tree_method="hist"
            )

    def build_pipeline(self, **model_kwargs) -> Pipeline:
        """Constructs end-to-end scikit-learn Pipeline with preprocessing and estimator."""
        preprocessor = self.feature_engineer.build_preprocessing_pipeline()
        estimator = self._get_base_estimator(**model_kwargs)

        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator)
        ])
        return self.pipeline

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **model_kwargs) -> Pipeline:
        """
        Trains regression pipeline on training features.

        Args:
            X_train: Training features DataFrame.
            y_train: Continuous target delivery duration in minutes.

        Returns:
            Fitted scikit-learn Pipeline.
        """
        logger.info(f"Training ETA Regressor using algorithm: {self.algorithm}")
        self.build_pipeline(**model_kwargs)
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        logger.info("ETA Regressor training completed.")
        return self.pipeline

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates continuous delivery duration predictions (in minutes)."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call train() or load_model() before predict().")
        return self.pipeline.predict(X)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = config.MODEL_CONFIG.cv_folds,
        scoring: str = "neg_root_mean_squared_error"
    ) -> Dict[str, float]:
        """Performs K-fold cross-validation."""
        logger.info(f"Executing {cv}-fold cross-validation with scoring: {scoring}")
        pipeline = self.build_pipeline()
        kf = KFold(n_splits=cv, shuffle=True, random_state=config.MODEL_CONFIG.random_state)
        scores = cross_val_score(pipeline, X, y, cv=kf, scoring=scoring, n_jobs=config.MODEL_CONFIG.n_jobs)

        return {
            "cv_mean_score": float(np.mean(scores)),
            "cv_std_score": float(np.std(scores)),
            "cv_scores": [float(s) for s in scores]
        }

    def explain_with_shap(self, X_sample: pd.DataFrame) -> Any:
        """
        Calculates SHAP explainability values for feature attribution.

        Args:
            X_sample: Input feature sample for SHAP value computation.

        Returns:
            SHAP explanation object.
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model must be trained before computing SHAP explanations.")

        import shap
        regressor = self.pipeline.named_steps["regressor"]
        preprocessor = self.pipeline.named_steps["preprocessor"]

        X_transformed = preprocessor.transform(X_sample)

        if self.algorithm in ["xgboost", "random_forest"]:
            explainer = shap.TreeExplainer(regressor)
            shap_values = explainer(X_transformed)
        else:
            explainer = shap.Explainer(regressor, X_transformed)
            shap_values = explainer(X_transformed)

        return shap_values

    def save_model(self, filepath: Optional[Path] = None) -> Path:
        """Serializes trained pipeline to disk."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Cannot save unfitted model.")

        save_path = Path(filepath) if filepath else config.MODELS_DIR / f"eta_regressor_{self.algorithm}.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, save_path)
        logger.info(f"Saved trained regression model to: {save_path}")
        return save_path

    def load_model(self, filepath: Path) -> Pipeline:
        """Loads serialized pipeline from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at: {filepath}")

        self.pipeline = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded regression model from: {filepath}")
        return self.pipeline

