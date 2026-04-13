from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd


def calculate_summary_metrics(customer_df: pd.DataFrame) -> Dict[str, Any]:
    required_cols = {"CustomerID", "Monetary", "Segment", "ChurnRisk"}
    missing = required_cols - set(customer_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for reporting: {missing}")

    total_customers = int(customer_df["CustomerID"].nunique())
    total_revenue = float(customer_df["Monetary"].sum())
    avg_customer_revenue = float(customer_df["Monetary"].mean())
    churn_risk_customers = int(customer_df["ChurnRisk"].sum())

    vip_revenue = float(
        customer_df.loc[customer_df["Segment"] == "VIP Customers", "Monetary"].sum()
    )
    vip_revenue_share_pct = (
        (vip_revenue / total_revenue) * 100 if total_revenue > 0 else 0.0
    )

    return {
        "total_customers": total_customers,
        "total_revenue": round(total_revenue, 2),
        "avg_customer_revenue": round(avg_customer_revenue, 2),
        "churn_risk_customers": churn_risk_customers,
        "vip_revenue_share_pct": round(vip_revenue_share_pct, 2),
    }


def build_cluster_profile(customer_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "Segment",
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "ChurnRisk",
    }
    missing = required_cols - set(customer_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for cluster profile: {missing}")

    profile = (
        customer_df.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            TotalRevenue=("Monetary", "sum"),
            ChurnRiskCustomers=("ChurnRisk", "sum"),
        )
        .reset_index()
    )

    total_revenue = profile["TotalRevenue"].sum()
    profile["RevenueSharePct"] = (
        profile["TotalRevenue"] / total_revenue * 100 if total_revenue > 0 else 0.0
    )

    numeric_cols = [
        "AvgRecency",
        "AvgFrequency",
        "AvgMonetary",
        "TotalRevenue",
        "RevenueSharePct",
    ]
    profile[numeric_cols] = profile[numeric_cols].round(2)

    desired_order = [
        "VIP Customers",
        "Loyal Customers",
        "Low Value Customers",
        "Inactive Customers",
    ]
    profile["Segment"] = pd.Categorical(
        profile["Segment"],
        categories=desired_order,
        ordered=True,
    )
    profile = profile.sort_values("Segment").reset_index(drop=True)

    return profile


def save_summary_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def print_summary_metrics(metrics: Dict[str, Any]) -> None:
    print(f"Total customers      : {metrics['total_customers']}")
    print(f"Total revenue        : {metrics['total_revenue']}")
    print(f"Avg customer revenue : {metrics['avg_customer_revenue']}")
    print(f"Churn-risk customers : {metrics['churn_risk_customers']}")
    print(f"VIP revenue share %  : {metrics['vip_revenue_share_pct']}")


def generate_visual_reports(
    customer_df: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    output_dir: str | Path,
    selected_k: int = 4,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Customers by Segment
    plt.figure(figsize=(10, 6))
    plt.bar(cluster_profile["Segment"].astype(str), cluster_profile["Customers"])
    plt.title("Customers by Segment")
    plt.xlabel("Segment")
    plt.ylabel("Customer Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / "customers_by_segment.png", dpi=200)
    plt.close()

    # 2. Revenue by Segment (REAL business revenue)
    plt.figure(figsize=(10, 6))
    plt.bar(cluster_profile["Segment"].astype(str), cluster_profile["TotalRevenue"])
    plt.title("Revenue by Segment")
    plt.xlabel("Segment")
    plt.ylabel("Total Revenue")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / "revenue_by_segment.png", dpi=200)
    plt.close()

    # 3. Revenue Share by Segment
    plt.figure(figsize=(10, 6))
    plt.bar(cluster_profile["Segment"].astype(str), cluster_profile["RevenueSharePct"])
    plt.title("Revenue Share by Segment")
    plt.xlabel("Segment")
    plt.ylabel("Revenue Share (%)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / "revenue_share_by_segment.png", dpi=200)
    plt.close()

    # 4. K selection - inertia
    plt.figure(figsize=(9, 5))
    plt.plot(evaluation_df["k"], evaluation_df["inertia"], marker="o")
    plt.axvline(selected_k, linestyle="--")
    plt.title("K Selection: Inertia Trend")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(output_dir / "k_inertia.png", dpi=200)
    plt.close()

    # 5. K selection - silhouette
    plt.figure(figsize=(9, 5))
    plt.plot(evaluation_df["k"], evaluation_df["silhouette_score"], marker="o")
    plt.axvline(selected_k, linestyle="--")
    plt.title("K Selection: Silhouette Score")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(output_dir / "k_silhouette.png", dpi=200)
    plt.close()

    # 6. Recency vs Monetary scatter
    plt.figure(figsize=(12, 7))
    for segment in [
        "VIP Customers",
        "Loyal Customers",
        "Low Value Customers",
        "Inactive Customers",
    ]:
        subset = customer_df[customer_df["Segment"] == segment]
        plt.scatter(
            subset["Recency"],
            subset["Monetary"],
            s=20,
            alpha=0.5,
            label=segment,
        )

    plt.title("Customer Behavior: Recency vs Revenue")
    plt.xlabel("Recency (days)")
    plt.ylabel("Monetary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "recency_vs_revenue.png", dpi=200)
    plt.close()