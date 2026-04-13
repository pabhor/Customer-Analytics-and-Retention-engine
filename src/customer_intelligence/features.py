from __future__ import annotations

import pandas as pd
import numpy as np


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer-level features from cleaned transaction data.

    Expected columns in df:
    - CustomerID
    - InvoiceNo
    - InvoiceDate
    - Quantity
    - UnitPrice
    - StockCode

    Returns:
        customer_df with ORIGINAL business metrics preserved.
    """
    required_cols = {
        "CustomerID",
        "InvoiceNo",
        "InvoiceDate",
        "Quantity",
        "UnitPrice",
        "StockCode",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    work_df = df.copy()

    # Ensure datetime
    work_df["InvoiceDate"] = pd.to_datetime(work_df["InvoiceDate"], errors="coerce")

    # Revenue per row
    work_df["LineRevenue"] = work_df["Quantity"] * work_df["UnitPrice"]

    # Reference date for recency
    snapshot_date = work_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Customer-level aggregation
    customer_df = (
        work_df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("LineRevenue", "sum"),
            TotalItems=("Quantity", "sum"),
            UniqueProducts=("StockCode", "nunique"),
            LastPurchaseDate=("InvoiceDate", "max"),
            FirstPurchaseDate=("InvoiceDate", "min"),
        )
        .reset_index()
    )

    # Purchase span
    customer_df["PurchaseSpanDays"] = (
        customer_df["LastPurchaseDate"] - customer_df["FirstPurchaseDate"]
    ).dt.days

    # Average basket value
    customer_df["AvgBasketValue"] = (
        customer_df["Monetary"] / customer_df["Frequency"].replace(0, np.nan)
    ).fillna(0.0)

    # Average unit price proxy
    customer_df["AvgUnitPrice"] = (
        customer_df["Monetary"] / customer_df["TotalItems"].replace(0, np.nan)
    ).fillna(0.0)

    # Average days between purchases
    customer_df["AvgDaysBetweenPurchases"] = np.where(
        customer_df["Frequency"] > 1,
        customer_df["PurchaseSpanDays"] / (customer_df["Frequency"] - 1),
        customer_df["Recency"],
    )

    # Keep only valid non-negative business values
    numeric_cols = [
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "UniqueProducts",
        "AvgBasketValue",
        "AvgUnitPrice",
        "PurchaseSpanDays",
        "AvgDaysBetweenPurchases",
    ]

    for col in numeric_cols:
        customer_df[col] = pd.to_numeric(customer_df[col], errors="coerce").fillna(0)

    # IMPORTANT:
    # Do NOT overwrite Monetary/Frequency/etc. with logs here.
    # Original business columns must remain intact for reporting.

    return customer_df