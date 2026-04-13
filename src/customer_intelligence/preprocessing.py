from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessingArtifacts:
    scaler: StandardScaler
    feature_columns: list[str]
    clip_bounds: dict[str, tuple[float, float]]
    log_transform_features: list[str]


class FeaturePreprocessor:
    def __init__(
        self,
        feature_columns: list[str],
        log_transform_features: list[str],
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        self.feature_columns = feature_columns
        self.log_transform_features = log_transform_features
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.scaler = StandardScaler()
        self.clip_bounds: dict[str, tuple[float, float]] = {}

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, PreprocessingArtifacts]:
        processed = self._prepare_features(df, fit=True)
        scaled = self.scaler.fit_transform(processed[self.feature_columns])
        artifacts = PreprocessingArtifacts(
            scaler=self.scaler,
            feature_columns=self.feature_columns,
            clip_bounds=self.clip_bounds,
            log_transform_features=self.log_transform_features,
        )
        return scaled, processed, artifacts

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        processed = self._prepare_features(df, fit=False)
        scaled = self.scaler.transform(processed[self.feature_columns])
        return scaled, processed

    def _prepare_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        prepared = df.copy()
        prepared[self.feature_columns] = prepared[self.feature_columns].astype(float)

        for col in self.feature_columns:
            if fit:
                lower = float(prepared[col].quantile(self.lower_quantile))
                upper = float(prepared[col].quantile(self.upper_quantile))
                self.clip_bounds[col] = (lower, upper)
            lower, upper = self.clip_bounds[col]
            prepared[col] = prepared[col].clip(lower=lower, upper=upper)

        for col in self.log_transform_features:
            if col in prepared.columns:
                prepared[col] = np.log1p(prepared[col])

        return prepared
