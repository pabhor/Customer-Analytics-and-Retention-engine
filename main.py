from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from customer_intelligence.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the customer intelligence pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to online_retail.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where outputs will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    customer_df = run_pipeline(
        input_csv=args.input,
        output_dir=args.output_dir,
    )

    print("\nPipeline completed successfully.")
    print(f"Final customer records: {len(customer_df)}")
    print(f"Unique customers      : {customer_df['CustomerID'].nunique()}")
    print(f"Segments created      : {customer_df['Segment'].nunique()}")
    print("\nArtifacts saved under outputs/\n")


if __name__ == "__main__":
    main()