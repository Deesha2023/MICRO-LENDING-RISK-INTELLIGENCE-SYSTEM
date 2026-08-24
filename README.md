# Loan Risk & Micro-Lending FinTech Project

## Database already available

This project is configured for the database and table supplied by the user:

- Database: `loan_database`
- Table: `loan_sample`

The uploaded CSV contains **150,000 records and 143 columns**. The user reports that only **83,000 records were successfully imported into MySQL**. This discrepancy is deliberately NOT hidden or corrected by the project.

The application uses the MySQL table as the operational source of truth. A data-audit command reports the actual database row count and target distribution. The CSV is kept outside the runtime workflow unless you explicitly want to compare it.

## What the project does

1. Connects to MySQL.
2. Reads `loan_sample`.
3. Validates required columns.
4. Creates a leakage-controlled modeling dataset.
5. Performs EDA and generates interactive HTML charts.
6. Trains Logistic Regression and Random Forest models.
7. Selects the better model using ROC-AUC.
8. Saves the trained model.
9. Calculates default probability and a project risk score.
10. Applies transparent eligibility rules.
11. Returns APPROVE / REVIEW / REJECT with reasons.
12. Provides a FastAPI endpoint.
13. Provides a Streamlit UI.

## Important: no hard-coded MySQL password

Copy `.env.example` to `.env` and enter your own MySQL password.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`.

## 1. Check database connection

```powershell
python -m src.db_check
```

This prints:
- connection status
- database/table name
- actual row count in MySQL
- columns
- `loan_condition` distribution
- key missing-value percentages

Do this first.

## 2. Run EDA

```powershell
python -m src.eda
```

Outputs are written to `reports/generated/`.

## 3. Train the model

```powershell
python -m src.train_model
```

The model uses `loan_condition` as the target:

- `good loan` -> 0
- `bad loan` -> 1

The project uses only pre-origination / application-time variables for modeling. Post-origination repayment fields and the target are excluded to reduce leakage.

Outputs:
- `models/loan_default_model.joblib`
- `models/model_metadata.json`
- `reports/generated/model_metrics.json`

## 4. Test a borrower assessment

```powershell
python -m src.assess_demo
```

## 5. Run FastAPI

```powershell
uvicorn src.api:app --reload
```

Open:

`http://127.0.0.1:8000/docs`

Main endpoint:

`POST /assess`

## 6. Run Streamlit

Open another PowerShell window, activate the same environment, then:

```powershell
streamlit run app.py
```

Open:

`http://localhost:8501`

## Model and decision design

### Risk score

The model outputs probability of bad loan/default. The project converts this into a simple 0-100 score:

`risk_score = round(100 * (1 - bad_loan_probability))`

Higher score = safer.

### Risk categories

- 75-100: Low Risk
- 55-74: Medium Risk
- 35-54: High Risk
- 0-34: Very High Risk

These are project-defined categories, not a regulated credit score.

### Eligibility

The project deliberately labels its rules as configurable demonstration rules. They are not claimed to be legal/regulatory lending standards.

Default rules:
- annual income >= 24000
- FICO >= 600
- DTI <= 45
- revol_util <= 90
- loan-to-income <= 50%
- predicted bad-loan probability <= 45%

### Decision

- APPROVE: eligible + Low Risk
- REVIEW: eligible + Medium/High Risk
- REJECT: not eligible or Very High Risk

The engine returns reasons for every failed rule.

## Critical data note

The uploaded CSV has 150,000 rows. The MySQL table reportedly contains only about 83,000 imported rows. This project does not assume that the missing 67,000 records exist in MySQL. All model training and dashboard statistics are based on the records actually available in `loan_database.loan_sample`.

If you later re-import the missing records, simply rerun:

```powershell
python -m src.db_check
python -m src.eda
python -m src.train_model
```

No code changes are required.

## Expected columns

The uploaded CSV was inspected and contains 143 columns. The project primarily uses:

`loan_amnt, funded_amnt, funded_amnt_inv, term, int_rate, installment, grade, sub_grade, emp_length, home_ownership, annual_inc, verification_status, purpose, dti, delinq_2yrs, earliest_cr_line, inq_last_6mths, open_acc, pub_rec, revol_bal, revol_util, total_acc, application_type, acc_now_delinq, tot_coll_amt, tot_cur_bal, open_acc_6m, open_act_il, open_il_12m, open_il_24m, total_bal_il, open_rv_12m, open_rv_24m, total_rev_hi_lim, inq_last_12m, acc_open_past_24mths, avg_cur_bal, bc_open_to_buy, bc_util, mort_acc, loan_condition`

The target `loan_condition` is present in the uploaded CSV.

## If MySQL has only 83K rows

That is acceptable for this project. The software will train on the rows returned from MySQL. The exact number is printed before training.

## Troubleshooting

### Access denied
Check `MYSQL_USER` and `MYSQL_PASSWORD` in `.env`.

### Can't connect to MySQL
Make sure MySQL Server is running and port 3306 is correct.

### Unknown database
Confirm that `loan_database` exists.

### Unknown table
Confirm that `loan_sample` exists.

### Missing column
Run `python -m src.db_check`. The script reports exactly which required columns are missing.

### Model not found
Run `python -m src.train_model` before starting Streamlit or FastAPI.

## Project files

- `app.py` - Streamlit application
- `src/db.py` - database access
- `src/db_check.py` - connection and data audit
- `src/features.py` - feature engineering and preprocessing
- `src/eda.py` - EDA
- `src/train_model.py` - model training
- `src/risk_engine.py` - scoring, eligibility and decision
- `src/api.py` - FastAPI backend
- `src/assess_demo.py` - command-line demo
- `sql/analysis_queries.sql` - useful SQL queries


## Updated ML models and decision policy

The training pipeline now trains and evaluates:
- Logistic Regression
- Random Forest
- Decision Tree

For every model it prints and saves a confusion matrix. PNG and CSV confusion-matrix files are written to `reports/`, while all three trained pipelines are saved in `models/`.

Run:
```bash
python -m src.train_model
streamlit run app.py
```

Decision policy requested for this project:
- Low Risk → Loan Approved
- Medium Risk → Forward to Manager / Manual Decision
- High Risk → Loan Approved

The Streamlit dashboard also shows model comparison metrics and the confusion matrices for all three models.
