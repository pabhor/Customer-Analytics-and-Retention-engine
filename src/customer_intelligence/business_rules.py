from __future__ import annotations

import numpy as np
import pandas as pd


def apply_business_rules(customer_df: pd.DataFrame) -> pd.DataFrame:
    enriched = customer_df.copy()

    recency_threshold = float(enriched["Recency"].quantile(0.75))
    low_frequency_threshold = float(enriched["Frequency"].quantile(0.35))
    low_spend_threshold = float(enriched["Monetary"].quantile(0.35))

    enriched["ChurnRiskFlag"] = np.where(
        (enriched["Recency"] >= recency_threshold)
        & (
            (enriched["Frequency"] <= low_frequency_threshold)
            | (enriched["Monetary"] <= low_spend_threshold)
            | (enriched["Segment"] == "Inactive Customers")
        ),
        1,
        0,
    )

    enriched["ChurnRiskLabel"] = np.where(enriched["ChurnRiskFlag"] == 1, "Churn Risk", "Stable")

    enriched["RetentionAction"] = enriched["Segment"].map(
        {
            "VIP Customers": "Protect with exclusive rewards and premium retention offers",
            "Loyal Customers": "Grow basket size through cross-sell and loyalty campaigns",
            "Low Value Customers": "Nurture with low-cost promotions and product discovery",
            "Inactive Customers": "Run win-back offers and reactivation campaigns",
        }
    )

    return enriched


def build_executive_summary(customer_df: pd.DataFrame) -> dict[str, float | int]:
    total_customers = int(customer_df["CustomerID"].nunique())
    total_revenue = float(customer_df["Monetary"].sum())
    churn_risk_customers = int(customer_df["ChurnRiskFlag"].sum())
    average_customer_revenue = float(customer_df["Monetary"].mean())

    vip = customer_df[customer_df["Segment"] == "VIP Customers"]
    vip_revenue_share = float(100 * vip["Monetary"].sum() / total_revenue) if total_revenue else 0.0

    return {
        "total_customers": total_customers,
        "total_revenue": round(total_revenue, 2),
        "average_customer_revenue": round(average_customer_revenue, 2),
        "churn_risk_customers": churn_risk_customers,
        "vip_revenue_share_pct": round(vip_revenue_share, 2),
    }
