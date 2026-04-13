from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    return pd.read_csv(path, encoding="ISO-8859-1")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw data: {missing}")

    clean_df = df.copy()

    clean_df["InvoiceDate"] = pd.to_datetime(clean_df["InvoiceDate"], errors="coerce")

    # Remove rows with missing customer
    clean_df = clean_df.dropna(subset=["CustomerID"])

    # Standardize type
    clean_df["CustomerID"] = clean_df["CustomerID"].astype(int)

    # Remove cancelled invoices
    clean_df = clean_df[~clean_df["InvoiceNo"].astype(str).str.startswith("C")]

    # Keep only positive sales
    clean_df = clean_df[(clean_df["Quantity"] > 0) & (clean_df["UnitPrice"] > 0)]

    # Remove duplicates
    clean_df = clean_df.drop_duplicates()

    # Drop bad dates
    clean_df = clean_df.dropna(subset=["InvoiceDate"])

    return clean_df.reset_index(drop=True)