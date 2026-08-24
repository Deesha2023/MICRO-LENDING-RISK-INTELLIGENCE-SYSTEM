"""Run management-focused portfolio queries and export their results."""

from pathlib import Path
import re
import pandas as pd
from sqlalchemy import text

from .db import get_config, get_engine, quote_identifier, table_columns

OUT = Path("reports/generated/business_queries")


def _rate_expression():
    return (
        "100.0 * AVG(CASE WHEN LOWER(TRIM(`loan_condition`)) = "
        "'bad loan' THEN 1 ELSE 0 END)"
    )


def build_queries(columns):
    """Return only the queries supported by the imported table schema."""
    cols = set(columns)
    table = quote_identifier(get_config()["table"])
    rate = _rate_expression()
    queries = {
        "01_portfolio_overview": f"""
            SELECT COUNT(*) AS total_loans,
                   ROUND(SUM(`loan_amnt`), 2) AS total_amount,
                   ROUND(AVG(`loan_amnt`), 2) AS average_loan_amount,
                   ROUND({rate}, 2) AS bad_loan_rate_pct
            FROM {table}
        """ if {"loan_condition", "loan_amnt"} <= cols else None,
        "02_risk_by_grade": f"""
            SELECT `grade`, COUNT(*) AS loans,
                   ROUND(AVG(`loan_amnt`), 2) AS average_loan_amount,
                   ROUND({rate}, 2) AS bad_loan_rate_pct
            FROM {table}
            GROUP BY `grade` ORDER BY bad_loan_rate_pct DESC
        """ if {"loan_condition", "loan_amnt", "grade"} <= cols else None,
        "03_risk_by_purpose": f"""
            SELECT `purpose`, COUNT(*) AS loans,
                   ROUND(SUM(`loan_amnt`), 2) AS total_amount,
                   ROUND({rate}, 2) AS bad_loan_rate_pct
            FROM {table}
            GROUP BY `purpose` ORDER BY bad_loan_rate_pct DESC
        """ if {"loan_condition", "loan_amnt", "purpose"} <= cols else None,
        "04_risk_by_state": f"""
            SELECT `addr_state`, COUNT(*) AS loans,
                   ROUND({rate}, 2) AS bad_loan_rate_pct
            FROM {table}
            GROUP BY `addr_state` HAVING COUNT(*) >= 25
            ORDER BY bad_loan_rate_pct DESC
        """ if {"loan_condition", "addr_state"} <= cols else None,
        "05_interest_rate_by_outcome": f"""
            SELECT `loan_condition`, COUNT(*) AS loans,
                   ROUND(AVG(CAST(REPLACE(`int_rate`, '%', '') AS DECIMAL(10,2))), 2)
                       AS average_interest_rate_pct
            FROM {table} GROUP BY `loan_condition`
        """ if {"loan_condition", "int_rate"} <= cols else None,
        "06_dti_risk_bands": f"""
            SELECT CASE WHEN `dti` < 10 THEN 'Low (<10)'
                        WHEN `dti` < 20 THEN 'Medium (10-19.99)'
                        WHEN `dti` < 30 THEN 'High (20-29.99)'
                        ELSE 'Very High (30+)' END AS dti_band,
                   COUNT(*) AS loans, ROUND({rate}, 2) AS bad_loan_rate_pct
            FROM {table} WHERE `dti` IS NOT NULL
            GROUP BY dti_band ORDER BY MIN(`dti`)
        """ if {"loan_condition", "dti"} <= cols else None,
        "07_term_performance": f"""
            SELECT `term`, COUNT(*) AS loans,
                   ROUND(AVG(`loan_amnt`), 2) AS average_loan_amount,
                   ROUND({rate}, 2) AS bad_loan_rate_pct
            FROM {table} GROUP BY `term` ORDER BY bad_loan_rate_pct DESC
        """ if {"loan_condition", "loan_amnt", "term"} <= cols else None,
        "08_high_value_bad_loans": f"""
            SELECT `loan_amnt`, `annual_inc`, `int_rate`, `dti`, `grade`, `purpose`
            FROM {table}
            WHERE LOWER(TRIM(`loan_condition`)) = 'bad loan'
            ORDER BY `loan_amnt` DESC LIMIT 100
        """ if {"loan_condition", "loan_amnt", "annual_inc", "int_rate", "dti", "grade", "purpose"} <= cols else None,
    }
    return {name: re.sub(r"\s+", " ", sql).strip()
            for name, sql in queries.items() if sql}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    queries = build_queries(table_columns())
    if not queries:
        raise RuntimeError("No business queries match the columns in the MySQL table.")

    sections = ["<!doctype html><html><head><meta charset='utf-8'>",
                "<title>Business Query Report</title>",
                "<style>body{font-family:Arial;margin:32px}table{border-collapse:collapse;",
                "margin-bottom:32px}th,td{border:1px solid #ccc;padding:7px}",
                "th{background:#17365d;color:white}</style></head><body>",
                "<h1>Micro-Lending Business Query Report</h1>"]
    with get_engine().connect() as conn:
        for name, sql in queries.items():
            result = pd.read_sql(text(sql), conn)
            result.to_csv(OUT / f"{name}.csv", index=False)
            sections.extend([f"<h2>{name.replace('_', ' ').title()}</h2>",
                             result.to_html(index=False, border=0)])
    sections.append("</body></html>")
    (OUT / "BUSINESS_QUERY_REPORT.html").write_text(
        "".join(sections), encoding="utf-8"
    )
    print(f"Business queries complete. Files written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
