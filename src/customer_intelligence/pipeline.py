from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from joblib import dump

from .data import load_data, clean_transactions
from .features import build_customer_features
from .reporting import (
    build_cluster_profile,
    calculate_summary_metrics,
    generate_visual_reports,
    print_summary_metrics,
    save_summary_metrics,
)
from .segmentation import run_segmentation


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def run_pipeline(input_csv: str | Path, output_dir: str | Path = "outputs") -> pd.DataFrame:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)

    processed_dir = output_dir / "processed"
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load and clean
    raw_df = load_data(input_csv)
    clean_df = clean_transactions(raw_df)
    clean_df.to_csv(processed_dir / "cleaned_transactions.csv", index=False)

    # Build business features
    customer_df = build_customer_features(clean_df)
    customer_df.to_csv(processed_dir / "customer_features.csv", index=False)

    # Segmentation
    segmented_df, artifacts, cluster_name_map = run_segmentation(
        customer_df=customer_df,
        feature_cols=["Recency", "Frequency", "Monetary"],
        candidate_k=[2, 3, 4, 5, 6],
        random_state=42,
        n_init=20,
    )
    segmented_df.to_csv(processed_dir / "customer_segments.csv", index=False)

    # Reporting
    cluster_profile = build_cluster_profile(segmented_df)
    cluster_profile.to_csv(reports_dir / "cluster_profile.csv", index=False)

    summary_metrics = calculate_summary_metrics(segmented_df)
    save_summary_metrics(summary_metrics, reports_dir / "summary_metrics.json")
    print_summary_metrics(summary_metrics)

    evaluation_df = artifacts.evaluation_df.copy()
    evaluation_df.to_csv(reports_dir / "kmeans_evaluation.csv", index=False)

    generate_visual_reports(
        customer_df=segmented_df,
        cluster_profile=cluster_profile,
        evaluation_df=evaluation_df,
        output_dir=figures_dir,
        selected_k=4,
    )

    segmentation_metadata = {
        "selected_feature_columns": artifacts.feature_cols,
        "best_k_used": artifacts.best_k,
        "cluster_name_map": cluster_name_map,
        "scaler": "RobustScaler",
        "algorithm": "KMeans",
        "random_state": 42,
        "n_init": 20,
        "notes": [
            "Business KPIs are computed from original customer-level monetary values.",
            "Modeling uses a separate transformed matrix.",
            "RFM-based segmentation used for cleaner business interpretation.",
            "K candidates were evaluated using silhouette score, but final K=4 is enforced for business actionability.",
        ],
    }
    save_json(segmentation_metadata, reports_dir / "segmentation_metadata.json")

    # Save artifacts
    dump(artifacts.scaler, models_dir / "rfm_robust_scaler.joblib")
    dump(artifacts.kmeans, models_dir / "rfm_kmeans_model.joblib")

    return segmented_df