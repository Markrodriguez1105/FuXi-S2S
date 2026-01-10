from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class LinearBiasModel:
    """Simple linear correction: y = slope * x + intercept, with optional clipping."""

    slope: float
    intercept: float
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        y = self.slope * x + self.intercept
        if self.clip_min is not None:
            y = np.maximum(y, self.clip_min)
        if self.clip_max is not None:
            y = np.minimum(y, self.clip_max)
        return y


class BiasCorrector:
    """Fits per-variable linear bias correction models from paired obs/forecast samples."""

    def __init__(self):
        self.models: Dict[str, LinearBiasModel] = {}
        self.is_fitted: bool = False

    @staticmethod
    def _fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Least-squares fit for y = a*x + b."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 2:
            raise ValueError("Not enough valid samples to fit linear model")

        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(slope), float(intercept)

    def fit(
        self,
        training_df: pd.DataFrame,
        mapping: Optional[Dict[str, str]] = None,
    ) -> "BiasCorrector":
        """Fit correction models.

        Args:
            training_df: DataFrame containing both forecast and observation columns.
            mapping: Dict mapping forecast_col -> obs_col.
                     Defaults to: {'t2m_celsius':'TMAX','tp':'RAINFALL','wind_speed':'WINDSPEED'}
        """
        if mapping is None:
            mapping = {
                "t2m_celsius": "TMAX",
                "tp": "RAINFALL",
                "wind_speed": "WINDSPEED",
            }

        models: Dict[str, LinearBiasModel] = {}

        for forecast_col, obs_col in mapping.items():
            if forecast_col not in training_df.columns or obs_col not in training_df.columns:
                continue

            x = training_df[forecast_col].to_numpy(dtype=float)
            y = training_df[obs_col].to_numpy(dtype=float)
            slope, intercept = self._fit_linear(x, y)

            clip_min = None
            if forecast_col in {"tp", "wind_speed"}:
                clip_min = 0.0

            models[forecast_col] = LinearBiasModel(
                slope=slope,
                intercept=intercept,
                clip_min=clip_min,
            )

        if not models:
            raise ValueError(
                "No models fitted. Ensure training_df includes the required forecast/obs columns."
            )

        self.models = models
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted corrections and return a copy with new *_corrected columns."""
        if not self.is_fitted:
            raise RuntimeError("BiasCorrector is not fitted")

        out = df.copy()
        for forecast_col, model in self.models.items():
            if forecast_col not in out.columns:
                continue
            x = out[forecast_col].to_numpy(dtype=float)
            out[f"{forecast_col}_corrected"] = model.predict(x)
        return out

    def save(self, path: str | Path) -> str:
        """Save correction models to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "models": self.models,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> "BiasCorrector":
        """Load correction models from a pickle file."""
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)

        obj = cls()
        obj.models = payload.get("models", {})
        obj.is_fitted = bool(obj.models)
        return obj


def _list_csv_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*.csv") if p.is_file()])


def load_training_data(compare_result_dir: str | Path) -> Optional[pd.DataFrame]:
    """Load comparison CSVs (if they exist) as training data.

    This matches the intent of the workflow notebook section that calls:
        load_training_data('compare/result')

    If you don't have CSV comparison outputs, you can instead build training
    data directly from `output/` and `data/pagasa` in the training notebook.
    """
    files = _list_csv_files(compare_result_dir)
    if not files:
        return None

    dfs = []
    for p in files:
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            continue

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    return df


def train_and_save_corrector(
    training_df: pd.DataFrame,
    output_path: str | Path = "bias_correction_params.pkl",
    mapping: Optional[Dict[str, str]] = None,
) -> Tuple[BiasCorrector, str]:
    """Train a BiasCorrector and save it to disk."""
    corrector = BiasCorrector().fit(training_df, mapping=mapping)
    saved_path = corrector.save(output_path)
    return corrector, saved_path
