from pathlib import Path
from typing import Any, Dict, Optional

import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .db import load_loans
from .features import (
    create_model_input,
    get_model_features,
)
from .risk_engine import assess_borrower


# =========================================================
# PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT
    / "models"
    / "default_model.joblib"
)

METADATA_PATH = (
    ROOT
    / "models"
    / "model_metadata.json"
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Micro-Lending Risk Intelligence API",
    description=(
        "AI/ML based loan default risk assessment API "
        "using the loan_database.loan_sample MySQL table."
    ),
    version="2.0.0",
)


# =========================================================
# GLOBAL MODEL VARIABLES
# =========================================================

_model = None
_metadata = None


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    global _model
    global _metadata

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if not MODEL_PATH.exists():

        raise RuntimeError(
            "Trained model not found.\n\n"
            f"Expected location:\n{MODEL_PATH}\n\n"
            "Run:\n"
            "python -m src.train_model"
        )

    # -----------------------------------------------------
    # Check metadata
    # -----------------------------------------------------

    if not METADATA_PATH.exists():

        raise RuntimeError(
            "Model metadata not found.\n\n"
            f"Expected location:\n{METADATA_PATH}\n\n"
            "Run:\n"
            "python -m src.train_model"
        )

    # -----------------------------------------------------
    # Load model once
    # -----------------------------------------------------

    if _model is None:

        _model = joblib.load(
            MODEL_PATH
        )

    # -----------------------------------------------------
    # Load metadata once
    # -----------------------------------------------------

    if _metadata is None:

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            _metadata = json.load(
                file
            )

    return _model, _metadata


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():

    return {

        "project":
            "Micro-Lending Risk Intelligence API",

        "status":
            "running",

        "database":
            "loan_database",

        "table":
            "loan_sample",

        "documentation":
            "/docs",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    try:

        model, metadata = load_model()

        return {

            "status":
                "healthy",

            "model_loaded":
                model is not None,

            "model":
                metadata.get(
                    "best_model",
                    "unknown"
                ),

            "features":
                metadata.get(
                    "features",
                    []
                ),
        }

    except Exception as e:

        return {

            "status":
                "unhealthy",

            "model_loaded":
                False,

            "error":
                str(e),
        }


# =========================================================
# MODEL INFORMATION
# =========================================================

@app.get("/model-info")
def model_info():

    try:

        _, metadata = load_model()

        return {

            "database":
                metadata.get(
                    "database",
                    "loan_database"
                ),

            "table":
                metadata.get(
                    "table",
                    "loan_sample"
                ),

            "training_rows":
                metadata.get(
                    "training_rows"
                ),

            "best_model":
                metadata.get(
                    "best_model"
                ),

            "features":
                metadata.get(
                    "features",
                    []
                ),

            "numeric_features":
                metadata.get(
                    "numeric_features",
                    []
                ),

            "categorical_features":
                metadata.get(
                    "categorical_features",
                    []
                ),

            "metrics":
                metadata.get(
                    "model_metrics",
                    {}
                ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# BORROWER REQUEST MODEL
# =========================================================

class BorrowerRequest(BaseModel):

    # -----------------------------------------------------
    # Financial features
    # -----------------------------------------------------

    loan_amnt: Optional[float] = Field(
        default=None,
        description="Requested loan amount"
    )

    funded_amnt: Optional[float] = Field(
        default=None,
        description="Funded loan amount"
    )

    funded_amnt_inv: Optional[float] = Field(
        default=None,
        description="Investor funded amount"
    )

    int_rate: Optional[float] = Field(
        default=None,
        description="Interest rate"
    )

    installment: Optional[float] = Field(
        default=None,
        description="Monthly installment"
    )

    annual_inc: Optional[float] = Field(
        default=None,
        description="Annual income"
    )

    dti: Optional[float] = Field(
        default=None,
        description="Debt-to-income ratio"
    )

    revol_bal: Optional[float] = Field(
        default=None,
        description="Revolving balance"
    )

    revol_util: Optional[float] = Field(
        default=None,
        description="Revolving utilization"
    )

    # -----------------------------------------------------
    # Credit information
    # -----------------------------------------------------

    delinq_2yrs: Optional[float] = None

    inq_last_6mths: Optional[float] = None

    open_acc: Optional[float] = None

    pub_rec: Optional[float] = None

    total_acc: Optional[float] = None

    acc_now_delinq: Optional[float] = None

    tot_coll_amt: Optional[float] = None

    tot_cur_bal: Optional[float] = None

    open_acc_6m: Optional[float] = None

    open_act_il: Optional[float] = None

    open_il_12m: Optional[float] = None

    open_il_24m: Optional[float] = None

    total_bal_il: Optional[float] = None

    open_rv_12m: Optional[float] = None

    open_rv_24m: Optional[float] = None

    total_rev_hi_lim: Optional[float] = None

    inq_last_12m: Optional[float] = None

    acc_open_past_24mths: Optional[float] = None

    avg_cur_bal: Optional[float] = None

    bc_open_to_buy: Optional[float] = None

    bc_util: Optional[float] = None

    mort_acc: Optional[float] = None

    pub_rec_bankruptcies: Optional[float] = None

    tax_liens: Optional[float] = None

    # -----------------------------------------------------
    # Categorical features
    # -----------------------------------------------------

    term: Optional[str] = Field(
        default=None,
        description="Loan term"
    )

    grade: Optional[str] = Field(
        default=None,
        description="Loan grade"
    )

    sub_grade: Optional[str] = Field(
        default=None,
        description="Loan sub-grade"
    )

    emp_length: Optional[str] = Field(
        default=None,
        description="Employment length"
    )

    home_ownership: Optional[str] = Field(
        default=None,
        description="Home ownership"
    )

    verification_status: Optional[str] = Field(
        default=None,
        description="Verification status"
    )

    purpose: Optional[str] = Field(
        default=None,
        description="Loan purpose"
    )

    application_type: Optional[str] = Field(
        default=None,
        description="Application type"
    )

    # -----------------------------------------------------
    # Optional date fields
    # -----------------------------------------------------

    earliest_cr_line: Optional[str] = None

    issue_d: Optional[str] = None

    # -----------------------------------------------------
    # Optional FICO
    # -----------------------------------------------------

    fico_range_low: Optional[float] = None

    fico_range_high: Optional[float] = None

    # -----------------------------------------------------
    # Additional fields allowed
    # -----------------------------------------------------

    class Config:

        extra = "allow"


# =========================================================
# BORROWER TO DICTIONARY
# =========================================================

def borrower_to_dict(
    borrower: BorrowerRequest
) -> Dict[str, Any]:

    if hasattr(
        borrower,
        "model_dump"
    ):

        return borrower.model_dump(
            exclude_none=True
        )

    return borrower.dict(
        exclude_none=True
    )


# =========================================================
# RISK CATEGORY
# =========================================================

def calculate_risk_category(
    risk_score
):

    if risk_score >= 75:

        return "Low Risk"

    elif risk_score >= 55:

        return "Medium Risk"

    elif risk_score >= 35:

        return "High Risk"

    else:

        return "Very High Risk"


# =========================================================
# PREDICT BORROWER
# =========================================================

@app.post("/predict")
def predict(
    borrower: BorrowerRequest
):

    try:

        # -------------------------------------------------
        # Load model
        # -------------------------------------------------

        model, metadata = load_model()

        # -------------------------------------------------
        # Convert request
        # -------------------------------------------------

        borrower_data = borrower_to_dict(
            borrower
        )

        if not borrower_data:

            raise HTTPException(
                status_code=400,
                detail=(
                    "No borrower information "
                    "was provided."
                )
            )

        # -------------------------------------------------
        # Create model input
        # -------------------------------------------------

        X = create_model_input(
            borrower_data
        )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = int(
            model.predict(X)[0]
        )

        probability = float(
            model.predict_proba(X)[0][1]
        )

        # -------------------------------------------------
        # Risk score
        # -------------------------------------------------

        risk_score = round(
            (1 - probability) * 100,
            2
        )

        risk_category = (
            calculate_risk_category(
                risk_score
            )
        )

        # -------------------------------------------------
        # Risk engine
        # -------------------------------------------------

        assessment = assess_borrower(
            borrower_data,
            probability,
            risk_score
        )

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return {

            "success":
                True,

            "prediction": {

                "class":
                    prediction,

                "default_probability":
                    round(
                        probability,
                        6
                    ),

                "default_probability_pct":
                    round(
                        probability * 100,
                        2
                    ),
            },

            "risk": {

                "risk_score":
                    risk_score,

                "risk_category":
                    risk_category,
            },

            "decision": {

                "decision":
                    assessment.get(
                        "decision"
                    ),

                "eligible":
                    assessment.get(
                        "eligible"
                    ),

                "reasons":
                    assessment.get(
                        "reasons",
                        []
                    ),
            },

            "model":
                metadata.get(
                    "best_model",
                    "unknown"
                ),

            "features_used":
                metadata.get(
                    "features",
                    []
                ),
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# GET EXISTING BORROWER
# =========================================================

@app.get(
    "/borrower/{row_number}"
)
def get_borrower(
    row_number: int
):

    try:

        if row_number < 1:

            raise HTTPException(
                status_code=400,
                detail=(
                    "row_number must be "
                    "1 or greater."
                )
            )

        # -------------------------------------------------
        # Use model features
        # -------------------------------------------------

        features = get_model_features()

        # -------------------------------------------------
        # Load row
        # -------------------------------------------------

        df = load_loans(
            columns=features,
            limit=1,
            offset=row_number - 1
        )

        if df.empty:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Borrower row "
                    f"{row_number} "
                    "was not found."
                )
            )

        # -------------------------------------------------
        # Convert to dictionary
        # -------------------------------------------------

        record = (
            df.iloc[0]
            .where(
                pd.notna(
                    df.iloc[0]
                ),
                None
            )
            .to_dict()
        )

        return {

            "success":
                True,

            "row_number":
                row_number,

            "borrower":
                record,
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# PREDICT EXISTING BORROWER
# =========================================================

@app.post(
    "/predict-existing/{row_number}"
)
def predict_existing(
    row_number: int
):

    try:

        if row_number < 1:

            raise HTTPException(
                status_code=400,
                detail=(
                    "row_number must be "
                    "1 or greater."
                )
            )

        # -------------------------------------------------
        # Load model
        # -------------------------------------------------

        model, metadata = load_model()

        # -------------------------------------------------
        # Get features
        # -------------------------------------------------

        features = metadata.get(
            "features",
            get_model_features()
        )

        # -------------------------------------------------
        # Load borrower
        # -------------------------------------------------

        df = load_loans(
            columns=features,
            limit=1,
            offset=row_number - 1
        )

        if df.empty:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Borrower row "
                    f"{row_number} "
                    "was not found."
                )
            )

        borrower = (
            df.iloc[0]
            .where(
                pd.notna(
                    df.iloc[0]
                ),
                None
            )
            .to_dict()
        )

        # -------------------------------------------------
        # Prepare model input
        # -------------------------------------------------

        X = create_model_input(
            borrower
        )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = int(
            model.predict(X)[0]
        )

        probability = float(
            model.predict_proba(X)[0][1]
        )

        # -------------------------------------------------
        # Risk
        # -------------------------------------------------

        risk_score = round(
            (1 - probability) * 100,
            2
        )

        risk_category = (
            calculate_risk_category(
                risk_score
            )
        )

        # -------------------------------------------------
        # Assessment
        # -------------------------------------------------

        assessment = assess_borrower(
            borrower,
            probability,
            risk_score
        )

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return {

            "success":
                True,

            "row_number":
                row_number,

            "borrower":
                borrower,

            "prediction": {

                "class":
                    prediction,

                "default_probability":
                    round(
                        probability,
                        6
                    ),

                "default_probability_pct":
                    round(
                        probability * 100,
                        2
                    ),
            },

            "risk": {

                "risk_score":
                    risk_score,

                "risk_category":
                    risk_category,
            },

            "decision": {

                "decision":
                    assessment.get(
                        "decision"
                    ),

                "eligible":
                    assessment.get(
                        "eligible"
                    ),

                "reasons":
                    assessment.get(
                        "reasons",
                        []
                    ),
            },

            "model":
                metadata.get(
                    "best_model",
                    "unknown"
                ),
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# DATABASE SUMMARY
# =========================================================

@app.get(
    "/database-summary"
)
def database_summary():

    try:

        df = load_loans(
            columns=[
                "loan_condition"
            ]
        )

        if df.empty:

            return {

                "success":
                    True,

                "rows":
                    0,
            }

        condition = (
            df["loan_condition"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        good_loans = int(
            condition.eq(
                "good loan"
            ).sum()
        )

        bad_loans = int(
            condition.eq(
                "bad loan"
            ).sum()
        )

        total_rows = len(df)

        bad_rate = (

            bad_loans
            / total_rows
            * 100

            if total_rows > 0

            else 0
        )

        return {

            "success":
                True,

            "database":
                "loan_database",

            "table":
                "loan_sample",

            "rows":
                int(total_rows),

            "good_loans":
                good_loans,

            "bad_loans":
                bad_loans,

            "bad_loan_rate_pct":
                round(
                    bad_rate,
                    2
                ),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )