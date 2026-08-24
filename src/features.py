# ============================================================
# features.py
# Micro-Lending Risk Intelligence Project
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_METADATA_PATH = (
    ROOT / "models" / "model_metadata.json"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "loan_condition"


# ============================================================
# NUMERIC FEATURES
# ============================================================

NUMERIC_FEATURES = [

    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",

    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "acc_now_delinq",

    "tot_coll_amt",
    "tot_cur_bal",
    "open_acc_6m",
    "open_act_il",
    "open_il_12m",
    "open_il_24m",
    "total_bal_il",
    "open_rv_12m",
    "open_rv_24m",
    "total_rev_hi_lim",
    "inq_last_12m",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "mort_acc",
    "pub_rec_bankruptcies",
    "tax_liens",

    # Additional LendingClub features
    "num_accts_ever_120_pd",
    "mo_sin_old_rev_tl_op",
    "pct_tl_nvr_dlq",
    "num_op_rev_tl",
    "num_bc_sats",
    "num_tl_op_past_12m",
    "num_rev_tl_bal_gt_0",
    "delinq_amnt",
    "mths_since_recent_bc",
    "num_tl_30dpd",
    "num_actv_rev_tl",
    "num_tl_90g_dpd_24m",
    "percent_bc_gt_75",
    "num_rev_accts",
    "mths_since_last_major_derog",
    "mths_since_recent_inq",
    "num_actv_bc_tl",
    "num_bc_tl",
    "chargeoff_within_12_mths",
    "mo_sin_old_il_acct",
    "mths_since_last_delinq",
    "num_il_tl",
    "collections_12_mths_ex_med",
    "num_sats",

    # FICO
    "fico_range_low",
    "fico_range_high",
]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_FEATURES = [

    "term",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",

    # Additional categorical fields
    "addr_state",
    "initial_list_status",
]


# ============================================================
# OPTIONAL DATE FEATURES
# ============================================================

OPTIONAL_DATE_FEATURES = [
    "earliest_cr_line",
    "issue_d",
]


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

def clean_number(series):

    """
    Convert strings such as:

        10%
        1,000
        25.5

    into numeric values.

    Invalid values become numpy.nan.
    """

    result = pd.to_numeric(

        series
        .astype(str)
        .str.replace(
            "%",
            "",
            regex=False
        )
        .str.replace(
            ",",
            "",
            regex=False
        )
        .str.strip(),

        errors="coerce"
    )

    # IMPORTANT:
    # Make sure pandas nullable values are converted
    # to normal numpy NaN.

    return result.astype(float)


# ============================================================
# LOAD MODEL METADATA
# ============================================================

def load_metadata():

    if not MODEL_METADATA_PATH.exists():

        return {}

    try:

        with open(
            MODEL_METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return {}


# ============================================================
# GET MODEL FEATURES
# ============================================================

def get_model_features():

    """
    Return the exact feature list stored in the
    trained model metadata.
    """

    metadata = load_metadata()

    features = metadata.get(
        "features",
        []
    )

    if features:

        return list(
            dict.fromkeys(features)
        )

    return list(
        dict.fromkeys(
            NUMERIC_FEATURES
            + CATEGORICAL_FEATURES
            + [
                "credit_history_years",
                "issue_year",
                "loan_to_income_pct",
                "avg_fico",
            ]
        )
    )


# ============================================================
# ENGINEER FEATURES
# ============================================================

def engineer_features(df):

    x = df.copy()

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    for column in NUMERIC_FEATURES:

        if column in x.columns:

            x[column] = clean_number(
                x[column]
            )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        if column in x.columns:

            # IMPORTANT:
            # Do NOT use pandas StringDtype here.
            #
            # sklearn can encounter problems with pd.NA.
            # Use normal object dtype + numpy.nan instead.

            x[column] = (
                x[column]
                .astype(object)
                .replace(
                    [
                        "nan",
                        "NaN",
                        "None",
                        "",
                        "NA",
                        "N/A",
                    ],
                    np.nan
                )
            )

    # --------------------------------------------------------
    # Loan term
    # --------------------------------------------------------

    if "term" in x.columns:

        extracted = (
            x["term"]
            .astype(str)
            .str.extract(
                r"(\d+)",
                expand=False
            )
        )

        x["term"] = extracted.astype(object)

        x.loc[
            x["term"].isin(
                ["nan", "None"]
            ),
            "term"
        ] = np.nan

    # --------------------------------------------------------
    # Employment length
    # --------------------------------------------------------

    if "emp_length" in x.columns:

        extracted = (
            x["emp_length"]
            .astype(str)
            .str.extract(
                r"(\d+)",
                expand=False
            )
        )

        x["emp_length"] = extracted.astype(object)

        x.loc[
            x["emp_length"].isin(
                ["nan", "None"]
            ),
            "emp_length"
        ] = np.nan

    # --------------------------------------------------------
    # Credit history years
    # --------------------------------------------------------

    if "earliest_cr_line" in x.columns:

        dt = pd.to_datetime(
            x["earliest_cr_line"],
            errors="coerce"
        )

        current_year = (
            pd.Timestamp.today().year
        )

        x["credit_history_years"] = (

            current_year
            - dt.dt.year

        ).clip(
            lower=0
        )

        x["credit_history_years"] = (
            x["credit_history_years"]
            .astype(float)
        )

    # --------------------------------------------------------
    # Issue year
    # --------------------------------------------------------

    if "issue_d" in x.columns:

        dt = pd.to_datetime(
            x["issue_d"],
            errors="coerce"
        )

        x["issue_year"] = (
            dt.dt.year
            .astype(float)
        )

    # --------------------------------------------------------
    # Loan-to-income percentage
    # --------------------------------------------------------

    if (
        "annual_inc" in x.columns
        and "loan_amnt" in x.columns
    ):

        annual_income = clean_number(
            x["annual_inc"]
        )

        loan_amount = clean_number(
            x["loan_amnt"]
        )

        x["loan_to_income_pct"] = np.where(

            annual_income > 0,

            (
                loan_amount
                / annual_income
                * 100
            ),

            np.nan
        )

        x["loan_to_income_pct"] = (
            x["loan_to_income_pct"]
            .astype(float)
        )

    # --------------------------------------------------------
    # Average FICO
    # --------------------------------------------------------

    if (
        "fico_range_low" in x.columns
        and "fico_range_high" in x.columns
    ):

        fico_low = clean_number(
            x["fico_range_low"]
        )

        fico_high = clean_number(
            x["fico_range_high"]
        )

        x["avg_fico"] = (
            fico_low
            + fico_high
        ) / 2

        x["avg_fico"] = (
            x["avg_fico"]
            .astype(float)
        )

    return x


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_model_input(data):

    """
    Prepare borrower data for the already-trained model.

    Missing model columns are created as numpy.nan.

    IMPORTANT:
    All pandas pd.NA values are converted to numpy.nan.
    """

    # --------------------------------------------------------
    # Convert input to DataFrame
    # --------------------------------------------------------

    if isinstance(
        data,
        pd.DataFrame
    ):

        df = data.copy()

    elif isinstance(
        data,
        dict
    ):

        df = pd.DataFrame(
            [data]
        )

    else:

        raise TypeError(
            "Input must be a dictionary "
            "or pandas DataFrame."
        )

    # --------------------------------------------------------
    # Engineer features
    # --------------------------------------------------------

    df = engineer_features(
        df
    )

    # --------------------------------------------------------
    # Exact model features
    # --------------------------------------------------------

    model_features = (
        get_model_features()
    )

    # --------------------------------------------------------
    # Add missing columns
    # --------------------------------------------------------

    for column in model_features:

        if column not in df.columns:

            df[column] = np.nan

    # --------------------------------------------------------
    # Keep exact model columns
    # --------------------------------------------------------

    df = df[
        model_features
    ].copy()

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = load_metadata()

    metadata_numeric = metadata.get(
        "numeric_features",
        []
    )

    metadata_categorical = metadata.get(
        "categorical_features",
        []
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = list(
        dict.fromkeys(

            NUMERIC_FEATURES

            + list(metadata_numeric)

            + [
                "credit_history_years",
                "issue_year",
                "loan_to_income_pct",
                "avg_fico",
            ]
        )
    )

    for column in numeric_columns:

        if column in df.columns:

            df[column] = clean_number(
                df[column]
            )

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    categorical_columns = list(
        dict.fromkeys(

            CATEGORICAL_FEATURES

            + list(metadata_categorical)
        )
    )

    for column in categorical_columns:

        if column in df.columns:

            # IMPORTANT:
            # Convert to ordinary object dtype.
            # Replace ALL missing representations
            # with numpy.nan.

            df[column] = (
                df[column]
                .astype(object)
                .replace(
                    [
                        "nan",
                        "NaN",
                        "None",
                        "",
                        "NA",
                        "N/A",
                    ],
                    np.nan
                )
            )

            # Convert pandas NA values if any remain.

            df[column] = df[column].apply(
                lambda value:
                    np.nan
                    if pd.isna(value)
                    else value
            )

    # --------------------------------------------------------
    # FINAL GLOBAL NA CLEANUP
    # --------------------------------------------------------

    # This is the important part for your current error.
    #
    # Any remaining pd.NA is converted to np.nan.

    df = df.astype(
        object
    )

    df = df.where(
        pd.notna(df),
        np.nan
    )

    # --------------------------------------------------------
    # Final categorical cleanup
    # --------------------------------------------------------

    for column in categorical_columns:

        if column in df.columns:

            df[column] = df[column].apply(
                lambda value:
                    np.nan
                    if value is None
                    or pd.isna(value)
                    else str(value)
            )

    # --------------------------------------------------------
    # Final numeric cleanup
    # --------------------------------------------------------

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            ).astype(float)

    return df


# ============================================================
# AVAILABLE FEATURES
# ============================================================

def available_features(df):

    x = engineer_features(
        df
    )

    numeric_candidates = (

        NUMERIC_FEATURES

        + [
            "credit_history_years",
            "issue_year",
            "loan_to_income_pct",
            "avg_fico",
        ]
    )

    numeric = [

        column
        for column in numeric_candidates
        if column in x.columns
    ]

    categorical = [

        column
        for column in CATEGORICAL_FEATURES
        if column in x.columns
    ]

    return (
        x,
        numeric,
        categorical
    )


# ============================================================
# PREPROCESSOR
# ============================================================

def make_preprocessor(
    numeric,
    categorical
):

    transformers = []

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    if numeric:

        transformers.append(

            (
                "num",

                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        ),

                        (
                            "scaler",
                            StandardScaler()
                        ),
                    ]
                ),

                numeric
            )
        )

    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    if categorical:

        transformers.append(

            (
                "cat",

                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),

                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            )
                        ),
                    ]
                ),

                categorical
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# TARGET TO BINARY
# ============================================================

def target_to_binary(series):

    s = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {

        "good loan": 0,

        "bad loan": 1,
    }

    y = s.map(
        mapping
    )

    if y.isna().any():

        invalid = (
            s[y.isna()]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "loan_condition contains "
            "unexpected values: "
            f"{invalid}"
        )

    return y.astype(int)