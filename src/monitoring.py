"""Prediction logging + PSI drift detection."""

import math
import time
from collections import deque
from typing import List

import numpy as np

WINDOW = 1000  # rolling window of recent predictions


class Monitor:
    def __init__(self, baseline_means: List[float], baseline_stds: List[float]):
        self.baseline_means = np.array(baseline_means)
        self.baseline_stds = np.array(baseline_stds)
        self.latencies: deque = deque(maxlen=WINDOW)
        self.scores: deque = deque(maxlen=WINDOW)
        self.feature_sums: np.ndarray | None = None
        self.feature_sq_sums: np.ndarray | None = None
        self.n = 0
        self.started = time.time()

    def log(self, features: np.ndarray, score: float, latency_ms: float):
        self.latencies.append(latency_ms)
        self.scores.append(score)
        flat = features.ravel()
        if self.feature_sums is None:
            self.feature_sums = np.zeros_like(flat, dtype=float)
            self.feature_sq_sums = np.zeros_like(flat, dtype=float)
        self.feature_sums += flat
        self.feature_sq_sums += flat ** 2
        self.n += 1

    def latency_stats(self) -> dict:
        lats = sorted(self.latencies)
        if not lats:
            return {"p50_ms": 0, "p99_ms": 0, "count": 0}
        return {
            "p50_ms": round(lats[len(lats) // 2], 2),
            "p99_ms": round(lats[int(len(lats) * 0.99)], 2),
            "count": len(lats),
        }

    def psi(self) -> float:
        """Population Stability Index vs training baseline (mean-shift approx)."""
        if self.n < 100 or self.feature_sums is None:
            return 0.0
        current_means = self.feature_sums / self.n
        # binned PSI approximation on standardized shift
        shifts = np.abs(current_means - self.baseline_means) / np.maximum(self.baseline_stds, 1e-9)
        # map mean-shift to a pseudo-PSI
        return float(np.mean([s * 0.1 for s in shifts]))

    def drift_status(self) -> dict:
        psi = self.psi()
        return {
            "psi": round(psi, 4),
            "drift_detected": psi > 0.25,
            "n_observed": self.n,
        }
