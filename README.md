# Customer Intelligence Pipeline

Production-style Python project for retail customer segmentation, churn-risk identification, and retention strategy generation using the UCI Online Retail dataset.

## What this project does

- Cleans transactional retail data
- Builds customer-level behavioral features
- Selects and trains a KMeans segmentation model
- Assigns business-friendly segment labels
- Flags churn-risk customers with transparent business rules
- Exports CSVs, figures, JSON reports, and model artifacts

## Final segments

- VIP Customers
- Loyal Customers
- Low Value Customers
- Inactive Customers

## Project structure

```text
customer_intelligence_project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
│   ├── figures/
│   └── reports/
├── src/
│   └── customer_intelligence/
│       ├── __init__.py
│       ├── business_rules.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── pipeline.py
│       ├── preprocessing.py
│       ├── reporting.py
│       ├── segmentation.py
│       └── utils.py
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## Setup in VS Code

### 1. Create virtual environment

```bash
python -m venv .venv
```

### 2. Activate it

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Input file

Put your `online_retail.csv` anywhere you want and pass its path using `--input`.

Example:

```bash
python main.py --input "data/raw/online_retail.csv"
```

## Recommended local setup

```text
data/raw/online_retail.csv
```

## Output artifacts

### Processed data

- `data/processed/cleaned_transactions.csv`
- `data/processed/customer_features.csv`
- `data/processed/customer_segments.csv`
- `data/processed/cluster_profile.csv`

### Reports

- `outputs/reports/data_quality_report.json`
- `outputs/reports/executive_summary.json`
- `outputs/reports/final_model_metrics.json`
- `outputs/reports/k_selection_metrics.json`
- `outputs/reports/top_customers.csv`

### Figures

- `outputs/figures/k_inertia.png`
- `outputs/figures/k_silhouette.png`
- `outputs/figures/customers_by_segment.png`
- `outputs/figures/revenue_by_segment.png`
- `outputs/figures/recency_vs_revenue.png`

### Saved models

- `models/kmeans_model.joblib`
- `models/preprocessing_artifacts.joblib`

## Feature engineering

The pipeline creates these customer-level features:

- `Recency`
- `Frequency`
- `Monetary`
- `TotalItems`
- `UniqueProducts`
- `AvgUnitPrice`
- `AvgBasketValue`
- `AvgDaysBetweenPurchases`
- `PurchaseSpan`
- `RepeatPurchaseRate`

## Modeling approach

### Data cleaning

The pipeline removes:

- missing customer IDs
- duplicate rows
- canceled invoices
- non-positive quantities
- non-positive unit prices

### Preprocessing

- winsorizes numeric features at configurable quantiles
- applies `log1p` to skewed features
- standardizes final clustering features

### Segmentation

- evaluates K from 2 to 8
- exports inertia and silhouette metrics
- trains final KMeans with 4 clusters
- maps raw clusters into business labels using cluster profile logic



