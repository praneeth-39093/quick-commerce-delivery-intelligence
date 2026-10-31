"""
Delivery Optimization and Operational Recommendation Engine.
Translates model predictions (ETA and Breach Probabilities) into actionable dispatch rules,
dynamic SLA buffering, vehicle allocation recommendations, and dark store routing decisions.
"""

from typing import Dict, Any, Optional
import logging
import numpy as np

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DeliveryOptimizer:
    """Rules and heuristics engine for quick-commerce operational dispatch optimization."""

    # Buffer rules (in minutes) based on atmospheric and traffic impediments
    WEATHER_BUFFERS = {
        "Sunny": 0.0,
        "Cloudy": 2.0,
        "Windy": 3.0,
        "Sandstorms": 7.0,
        "Stormy": 8.0,
        "Fog": 6.0
    }

    TRAFFIC_BUFFERS = {
        "Low": 0.0,
        "Medium": 3.0,
        "High": 6.0,
        "Jam": 10.0
    }

    def compute_dynamic_eta(
        self,
        predicted_eta_min: float,
        weather: str,
        traffic: str,
        safety_margin_min: float = 3.0
    ) -> Dict[str, float]:
        """
        Calculates customer-facing promised SLA by adding dynamic environmental buffers.

        Args:
            predicted_eta_min: Machine learning raw predicted travel duration.
            weather: Current weather condition.
            traffic: Current road traffic density.
            safety_margin_min: Base operational safety pad.

        Returns:
            Dict containing buffer breakdown and final promised delivery window.
        """
        w_buffer = self.WEATHER_BUFFERS.get(weather, 2.0)
        t_buffer = self.TRAFFIC_BUFFERS.get(traffic, 3.0)

        dynamic_promised_eta = round(predicted_eta_min + w_buffer + t_buffer + safety_margin_min, 1)

        return {
            "predicted_eta_min": round(predicted_eta_min, 1),
            "weather_buffer_min": w_buffer,
            "traffic_buffer_min": t_buffer,
            "safety_margin_min": safety_margin_min,
            "promised_sla_eta_min": dynamic_promised_eta
        }

    def recommend_fleet_dispatch(
        self,
        distance_km: float,
        traffic: str,
        weather: str
    ) -> Dict[str, Any]:
        """
        Recommends optimal vehicle type and dispatch policy based on route friction.

        Args:
            distance_km: Haversine distance in km.
            traffic: Traffic density.
            weather: Weather condition.

        Returns:
            Dict containing recommended vehicle type and dispatch instructions.
        """
        if traffic == "Jam" and distance_km <= 3.5:
            rec_vehicle = "electric_scooter / bicycle"
            rationale = "Severe gridlock on short-distance delivery; 2-wheelers bypass bottlenecks fastest."
            batch_policy = "Single order only (do not batch)."

        elif weather in ["Stormy", "Sandstorms"]:
            rec_vehicle = "heavy motorcycle"
            rationale = "Severe weather requires high-stability vehicle with weather-sealed container."
            batch_policy = "Limit to maximum 1 batch delivery."

        elif distance_km > 7.0:
            rec_vehicle = "motorcycle"
            rationale = "Long-range delivery requires reliable high-speed motorized vehicle."
            batch_policy = "Corridor batching permissible if prep times match."

        else:
            rec_vehicle = "scooter / motorcycle"
            rationale = "Standard urban conditions; standard dispatch routing applied."
            batch_policy = "Standard batching up to 2 orders."

        return {
            "recommended_vehicle": rec_vehicle,
            "rationale": rationale,
            "batching_policy": batch_policy
        }

    def generate_risk_mitigation_plan(
        self,
        late_risk_probability: float,
        predicted_eta_min: float,
        prep_time_min: float
    ) -> Dict[str, Any]:
        """
        Generates operational mitigation actions if late-order breach risk is high.

        Args:
            late_risk_probability: Classification probability of exceeding SLA (0.0 to 1.0).
            predicted_eta_min: Predicted duration.
            prep_time_min: Estimated restaurant prep latency.

        Returns:
            Dict containing urgency level and operational actions.
        """
        if late_risk_probability >= 0.70:
            urgency = "CRITICAL"
            actions = [
                "Expedite merchant kitchen priority (rush ticket).",
                "Assign top-tier rated delivery partner currently in transit near restaurant.",
                "Disallow secondary order stacking / multi-deliveries for this rider.",
                "Send proactive delay notification with compensation token to customer."
            ]
        elif late_risk_probability >= 0.40:
            urgency = "MODERATE"
            actions = [
                "Monitor pickup timestamp; send automated ping to rider 5 mins before pickup.",
                "Verify kitchen order acceptance status."
            ]
        else:
            urgency = "LOW"
            actions = ["Standard operational dispatch flow."]

        return {
            "urgency_level": urgency,
            "late_probability_pct": round(late_risk_probability * 100, 1),
            "recommended_actions": actions
        }

