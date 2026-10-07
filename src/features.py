"""Feature engineering + Redis cache layer."""

import hashlib
import json
import os
from typing import Optional

import numpy as np

FEATURE_NAMES = [
    "amount",
    "amount_log",
    "hour_of_day",
    "is_night",
    "days_since_last",
    "avg_amount_30d",
    "amount_vs_avg",
    "merchant_risk",
]

MERCHANT_RISK = {
    "grocery": 0.1, "gas": 0.15, "restaurant": 0.2, "retail": 0.25,
    "travel": 0.4, "electronics": 0.45, "jewelry": 0.6, "crypto": 0.9,
    "unknown": 0.5,
}

_redis = None


def get_redis():
    global _redis
    if _redis is None:
        try:
            import redis
            _redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                socket_connect_timeout=1,
                decode_responses=True,
            )
            _redis.ping()
        except Exception:
            _redis = False  # gracefully degrade to no-cache
    return _redis if _redis else None


def engineer_features(event: dict) -> np.ndarray:
    """Raw event dict -> model feature vector."""
    amount = float(event["amount"])
    hour = int(event["hour_of_day"])
    avg30 = float(event.get("avg_amount_30d", amount))
    merchant = event.get("merchant_category", "unknown")

    return np.array([[
        amount,
        np.log1p(amount),
        hour,
        1.0 if (hour < 6 or hour >= 23) else 0.0,
        float(event.get("days_since_last", 0)),
        avg30,
        amount / max(avg30, 1.0),
        MERCHANT_RISK.get(merchant, 0.5),
    ]])


def cache_key(event: dict) -> str:
    stable = json.dumps(
        {k: event.get(k) for k in ("amount", "merchant_category", "hour_of_day")},
        sort_keys=True,
    )
    return "feat:" + hashlib.sha256(stable.encode()).hexdigest()[:16]


def get_cached_prediction(event: dict) -> Optional[float]:
    r = get_redis()
    if not r:
        return None
    val = r.get(cache_key(event))
    return float(val) if val is not None else None


def set_cached_prediction(event: dict, proba: float, ttl: int = 300):
    r = get_redis()
    if r:
        r.setex(cache_key(event), ttl, str(proba))
