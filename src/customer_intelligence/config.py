from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PipelineConfig:
    random_state: int = 42
    n_clusters: int = 4
    min_k: int = 2
    max_k: int = 6
    outlier_lower_quantile: float = 0.01
    outlier_upper_quantile: float = 0.99
    log_transform_features: Sequence[str] = field(
        default_factory=lambda: [
            "Frequency",
            "Monetary",
            "TotalItems",
            "UniqueProducts",
            "AvgBasketValue",
            "AvgDaysBetweenPurchases",
        ]
    )
    clustering_features: Sequence[str] = field(
        default_factory=lambda: [
            "Recency",
            "Frequency",
            "Monetary",
            "TotalItems",
            "UniqueProducts",
            "AvgBasketValue",
            "AvgDaysBetweenPurchases",
            "RepeatPurchaseRate",
        ]
    )
    top_n_customers: int = 10


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
