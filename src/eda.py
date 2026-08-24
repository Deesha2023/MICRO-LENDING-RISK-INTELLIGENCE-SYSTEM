from pathlib import Path
import json
import pandas as pd
import plotly.express as px
from .db import load_loans

OUT = Path("reports/generated")
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.write_html(
        OUT / f"{name}.html",
        include_plotlyjs="cdn"
    )


def clean_numeric(df, columns):
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False),
                errors="coerce"
            )


def main():

    # Only request columns that are actually present
    # in the database table.
    requested_cols = [
        "loan_condition",
        "loan_amnt",
        "annual_inc",
        "int_rate",
        "grade",
        "purpose",
        "dti",
        "revol_util",
        "term",
        "addr_state"
    ]

    df = load_loans(columns=requested_cols)

    if df.empty:
        raise RuntimeError(
            "No records were returned from loan_database.loan_sample."
        )

    # -----------------------------
    # Clean loan condition
    # -----------------------------
    if "loan_condition" not in df.columns:
        raise RuntimeError(
            "The required column 'loan_condition' does not exist "
            "in loan_database.loan_sample."
        )

    df["loan_condition"] = (
        df["loan_condition"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # -----------------------------
    # Numeric conversions
    # -----------------------------
    numeric_columns = [
        "loan_amnt",
        "annual_inc",
        "int_rate",
        "dti",
        "revol_util"
    ]

    clean_numeric(df, numeric_columns)

    # -----------------------------
    # Summary
    # -----------------------------
    summary = {
        "rows": int(len(df)),
        "columns_loaded": int(len(df.columns)),
        "bad_loan_rate_pct": float(
            df["loan_condition"]
            .eq("bad loan")
            .mean() * 100
        ),
        "avg_loan_amount": float(
            df["loan_amnt"].mean()
        ) if "loan_amnt" in df.columns else None,
        "avg_income": float(
            df["annual_inc"].mean()
        ) if "annual_inc" in df.columns else None
    }

    (OUT / "eda_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    # -----------------------------
    # 1. Good vs Bad Loans
    # -----------------------------
    save(
        px.histogram(
            df,
            x="loan_condition",
            title="Good vs Bad Loans"
        ),
        "01_loan_condition"
    )

    # -----------------------------
    # 2. DTI
    # -----------------------------
    if "dti" in df.columns:

        save(
            px.box(
                df,
                x="loan_condition",
                y="dti",
                title="DTI by Loan Condition"
            ),
            "02_dti"
        )

    # -----------------------------
    # 3. Revolving Utilization
    # -----------------------------
    if "revol_util" in df.columns:

        save(
            px.box(
                df,
                x="loan_condition",
                y="revol_util",
                title="Revolving Utilization by Loan Condition"
            ),
            "03_utilization"
        )

    # -----------------------------
    # 4. Grade Risk
    # -----------------------------
    if "grade" in df.columns:

        grade = (
            df.groupby(
                "grade",
                dropna=False
            )
            .agg(
                loans=("loan_condition", "size"),
                bad_rate=(
                    "loan_condition",
                    lambda s:
                    (s == "bad loan").mean() * 100
                )
            )
            .reset_index()
        )

        save(
            px.bar(
                grade,
                x="grade",
                y="bad_rate",
                text_auto=".2f",
                title="Bad Loan Rate by Grade"
            ),
            "04_grade_risk"
        )

    # -----------------------------
    # 5. Purpose Risk
    # -----------------------------
    if "purpose" in df.columns:

        purpose = (
            df.groupby(
                "purpose",
                dropna=False
            )
            .agg(
                loans=("loan_condition", "size"),
                bad_rate=(
                    "loan_condition",
                    lambda s:
                    (s == "bad loan").mean() * 100
                )
            )
            .reset_index()
            .sort_values(
                "bad_rate",
                ascending=False
            )
        )

        save(
            px.bar(
                purpose,
                x="purpose",
                y="bad_rate",
                text_auto=".2f",
                title="Bad Loan Rate by Purpose"
            ),
            "05_purpose_risk"
        )

    # -----------------------------
    # 6. Loan Amount
    # -----------------------------
    if "loan_amnt" in df.columns:

        save(
            px.box(
                df,
                x="loan_condition",
                y="loan_amnt",
                title="Loan Amount by Loan Condition"
            ),
            "06_loan_amount"
        )

    # -----------------------------
    # 7. Interest Rate
    # -----------------------------
    if "int_rate" in df.columns:

        save(
            px.box(
                df,
                x="loan_condition",
                y="int_rate",
                title="Interest Rate by Loan Condition"
            ),
            "07_interest_rate"
        )

    # -----------------------------
    # 8. Annual Income
    # -----------------------------
    if "annual_inc" in df.columns:

        save(
            px.box(
                df,
                x="loan_condition",
                y="annual_inc",
                title="Annual Income by Loan Condition"
            ),
            "08_income"
        )

    # -----------------------------
    # 9. State Risk
    # -----------------------------
    if "addr_state" in df.columns:

        state = (
            df.groupby(
                "addr_state",
                dropna=False
            )
            .agg(
                loans=("loan_condition", "size"),
                bad_rate=(
                    "loan_condition",
                    lambda s:
                    (s == "bad loan").mean() * 100
                )
            )
            .reset_index()
            .sort_values(
                "bad_rate",
                ascending=False
            )
        )

        save(
            px.bar(
                state,
                x="addr_state",
                y="bad_rate",
                title="Bad Loan Rate by State"
            ),
            "09_state_risk"
        )

    print(
        f"EDA complete. Files written to {OUT.resolve()}"
    )


if __name__ == "__main__":
    main()