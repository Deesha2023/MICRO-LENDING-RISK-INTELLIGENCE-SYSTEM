from pathlib import Path
import json
from .db import get_engine, get_config, table_count, table_columns, load_loans

REQUIRED = ["loan_condition", "loan_amnt", "annual_inc", "int_rate", "fico_range_low",
            "fico_range_high", "dti", "revol_util", "term", "grade", "purpose"]

def main():
    c = get_config()
    print(f"Database: {c['database']}")
    print(f"Table:    {c['table']}")
    engine = get_engine()
    with engine.connect() as conn:
        print("Connection: OK")
    n = table_count()
    print(f"MySQL row count: {n:,}")
    cols = table_columns()
    missing = [x for x in REQUIRED if x not in cols]
    print(f"Column count: {len(cols)}")
    if missing:
        print("MISSING REQUIRED COLUMNS:", missing)
        return
    print("Required columns: OK")
    df = load_loans(columns=["loan_condition"], limit=100000000)
    print("\nloan_condition:")
    print(df["loan_condition"].value_counts(dropna=False).to_string())
    print("\nTop missing-value percentages:")
    full = load_loans(columns=REQUIRED, limit=100000000)
    print((full.isna().mean().sort_values(ascending=False).head(15)*100).round(2).to_string())
    print("\nAudit completed.")

if __name__ == "__main__":
    main()
