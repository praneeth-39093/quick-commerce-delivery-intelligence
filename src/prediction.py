"""
Unified Inference Pipeline for Real-time and Batch Delivery Intelligence.
Combines data cleaning, feature engineering, ML models (or physics-informed heuristics if unfitted),
and operational optimization into a single caller-friendly interface.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
import numpy as np
import pandas as pd

import config
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.recommendations import DeliveryOptimizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DeliveryInferencePipeline:
    """Production inference service for real-time ETA and Late-Risk scoring."""

    def __init__(
        self,
        regressor_path: Optional[Path] = None,
        classifier_path: Optional[Path] = None
    ):
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.optimizer = DeliveryOptimizer()

        self.regressor = None
        self.classifier = None
        self.classifier_threshold = 0.50

        # Attempt to load trained models if available
        reg_candidates = [
            regressor_path,
            config.MODELS_DIR / "best_eta_regressor.joblib",
            config.MODELS_DIR / "eta_regressor_xgboost.joblib"
        ]
        reg_file = next((p for p in reg_candidates if p and p.exists()), None)

        clf_candidates = [
            classifier_path,
            config.MODELS_DIR / "best_late_classifier.joblib",
            config.MODELS_DIR / "late_classifier_xgboost.joblib"
        ]
        clf_file = next((p for p in clf_candidates if p and p.exists()), None)

        self._load_models_if_available(reg_file, clf_file)

    def _load_models_if_available(self, reg_file: Optional[Path], clf_file: Optional[Path]):
        """Loads serialized models if present on disk."""
        import joblib
        if reg_file and reg_file.exists():
            try:
                self.regressor = joblib.load(reg_file)
                logger.info(f"Loaded trained ETA regressor from {reg_file}")
            except Exception as e:
                logger.warning(f"Could not load regressor: {e}")

        if clf_file and clf_file.exists():
            try:
                loaded = joblib.load(clf_file)
                if isinstance(loaded, dict) and "threshold" in loaded:
                    self.classifier = loaded
                    self.classifier_threshold = loaded.get("threshold", 0.50)
                elif isinstance(loaded, dict) and "pipeline" in loaded:
                    self.classifier = loaded["pipeline"]
                    self.classifier_threshold = loaded.get("threshold", 0.50)
                else:
                    self.classifier = loaded
                logger.info(f"Loaded trained late-order classifier from {clf_file}")
            except Exception as e:
                logger.warning(f"Could not load classifier: {e}")

    def _heuristic_eta_fallback(self, row: pd.Series) -> float:
        """
        Physics-based domain heuristic for delivery ETA when models are not yet trained.
        Speed model based on vehicle, traffic, weather conditions, and order-placement attributes.
        NOTE: Does not require post-outcome 'Time_Order_picked'. Kitchen prep is estimated from order type.
        """
        dist = float(row.get("distance_km", 4.0))
        traffic = str(row.get("road_traffic_density", "Medium")).strip()
        weather = str(row.get("weather", "Sunny")).strip()
        order_type = str(row.get("type_of_order", "Meal")).strip()

        # Prep time estimated purely from order category at placement time
        category_prep_estimates = {"Snack": 10.0, "Drinks": 6.0, "Meal": 14.0, "Buffet": 20.0}
        prep = category_prep_estimates.get(order_type, 12.0)

        # Base speed in km/h based on traffic congestion
        speed_map = {"Low": 28.0, "Medium": 22.0, "High": 16.0, "Jam": 10.0}
        speed = speed_map.get(traffic, 20.0)

        # Weather speed discount
        if weather in ["Stormy", "Sandstorms", "Fog"]:
            speed *= 0.75
        elif weather in ["Windy", "Cloudy"]:
            speed *= 0.90

        transit_time_min = (dist / max(speed, 5.0)) * 60.0
        multiple = float(row.get("multiple_deliveries", 0))
        stacking_penalty = multiple * 6.0

        total_eta = prep + transit_time_min + stacking_penalty
        return round(float(total_eta), 1)

    def predict_single(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates comprehensive prediction and recommendation payload for a single order.
        Strictly leakage-free: operates solely on features available at order placement.

        Args:
            order_data: Dictionary of order attributes.

        Returns:
            Dict containing predicted ETA, late risk probability, promised SLA, and dispatch guidance.
        """
        df_single = pd.DataFrame([order_data])

        # Step 1: Cleaning
        df_cleaned = self.cleaner.clean(df_single, is_training=False)

        # Step 2: Feature Engineering
        df_featured = self.engineer.transform_features(df_cleaned, is_training=False)

        row = df_featured.iloc[0]

        # Step 3: ETA Prediction (ML model or baseline heuristic)
        # Guarantees X contains only order-placement features matching training pipeline
        if self.regressor is not None:
            X, _ = self.engineer.get_feature_matrices(df_featured, task="regression")
            if isinstance(self.regressor, dict) and "model" in self.regressor:
                X_proc = self.regressor["preprocessor"].transform(X)
                predicted_eta = float(self.regressor["model"].predict(X_proc)[0])
            else:
                predicted_eta = float(self.regressor.predict(X)[0])
            model_used_eta = "Trained ML Model"
        else:
            predicted_eta = self._heuristic_eta_fallback(row)
            model_used_eta = "Baseline Domain Heuristic"

        # Step 4: Late Delivery Risk (ML model or probability heuristic)
        if self.classifier is not None:
            X, _ = self.engineer.get_feature_matrices(df_featured, task="classification")
            if isinstance(self.classifier, dict) and "model" in self.classifier:
                X_proc = self.classifier["preprocessor"].transform(X)
                probs = self.classifier["model"].predict_proba(X_proc)[0]
            else:
                probs = self.classifier.predict_proba(X)[0]
            late_prob = float(probs[1]) if len(probs) > 1 else 0.5
            is_late_pred = int(late_prob >= self.classifier_threshold)
            model_used_clf = "Trained ML Model"
        else:
            # Calibrate risk against SLA threshold
            sla = config.DEFAULT_SLA_MINUTES
            diff = predicted_eta - sla
            # Sigmoid approximation of risk
            late_prob = float(1.0 / (1.0 + np.exp(-diff / 5.0)))
            is_late_pred = int(predicted_eta > sla)
            model_used_clf = "Baseline Domain Heuristic"

        # Step 5: Operational Recommendations
        weather = str(row.get("weather", "Sunny")).strip()
        traffic = str(row.get("road_traffic_density", "Medium")).strip()
        dist_km = float(row.get("distance_km", 3.5))

        order_type = str(row.get("type_of_order", "Meal")).strip()
        category_prep = {"Snack": 10.0, "Drinks": 6.0, "Meal": 14.0, "Buffet": 20.0}.get(order_type, 12.0)
        prep_min = float(row.get("prep_time_min", category_prep))
        if pd.isna(prep_min):
            prep_min = category_prep

        dynamic_sla = self.optimizer.compute_dynamic_eta(predicted_eta, weather, traffic)
        fleet_rec = self.optimizer.recommend_fleet_dispatch(dist_km, traffic, weather)
        risk_plan = self.optimizer.generate_risk_mitigation_plan(late_prob, predicted_eta, prep_min)

        return {
            "predicted_eta_minutes": round(predicted_eta, 1),
            "late_risk_probability": round(late_prob, 3),
            "is_late_predicted": is_late_pred,
            "eta_model_type": model_used_eta,
            "classifier_model_type": model_used_clf,
            "distance_km": round(dist_km, 2),
            "estimated_prep_time_min": round(prep_min, 1),
            "dynamic_sla": dynamic_sla,
            "fleet_recommendation": fleet_rec,
            "mitigation_plan": risk_plan
        }

    def predict_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Executes batch inference over multiple delivery records.

        Args:
            df_raw: Raw DataFrame of orders.

        Returns:
            pd.DataFrame: Augmented with predictions and risk scores.
        """
        df_cleaned = self.cleaner.clean(df_raw, is_training=False)
        df_featured = self.engineer.transform_features(df_cleaned, is_training=False)

        results = []
        for idx in range(len(df_featured)):
            row = df_featured.iloc[idx]
            if self.regressor is not None:
                X, _ = self.engineer.get_feature_matrices(df_featured.iloc[[idx]], task="regression")
                eta = float(self.regressor.predict(X)[0])
            else:
                eta = self._heuristic_eta_fallback(row)

            if self.classifier is not None:
                X, _ = self.engineer.get_feature_matrices(df_featured.iloc[[idx]], task="classification")
                prob = float(self.classifier.predict_proba(X)[0, 1])
            else:
                prob = float(1.0 / (1.0 + np.exp(-(eta - config.DEFAULT_SLA_MINUTES) / 5.0)))

            results.append({
                "predicted_eta_min": round(eta, 1),
                "late_risk_probability": round(prob, 3),
                "is_late_predicted": int(prob >= 0.5)
            })

        pred_df = pd.DataFrame(results)
        return pd.concat([df_featured.reset_index(drop=True), pred_df], axis=1)

