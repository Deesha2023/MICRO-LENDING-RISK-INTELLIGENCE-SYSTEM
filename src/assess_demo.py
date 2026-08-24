from pathlib import Path
import json

import joblib
import pandas as pd
from dotenv import load_dotenv

from .db import load_loans
from .risk_engine import assess_borrower, get_risk_category


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "default_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            "\nModel metadata not found.\n"
            "Please run:\n\n"
            "python -m src.train_model"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# CLEAN BORROWER DATA
# ============================================================

def clean_borrower_data(
    borrower_df,
    features
):

    df = borrower_df.copy()

    print("\nCleaning borrower data...")
    print("-" * 70)

    # --------------------------------------------------------
    # Convert empty strings and whitespace to NaN
    # --------------------------------------------------------

    for column in features:

        if column not in df.columns:
            continue

        if df[column].dtype == "object":

            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .replace(
                    {
                        "": pd.NA,
                        "nan": pd.NA,
                        "None": pd.NA,
                        "NULL": pd.NA,
                        "null": pd.NA,
                    }
                )
            )

    # --------------------------------------------------------
    # Try numeric conversion
    #
    # Only convert columns where all non-null values can
    # reasonably be interpreted as numbers.
    # --------------------------------------------------------

    for column in features:

        if column not in df.columns:
            continue

        if df[column].dtype == "object":

            original = df[column]

            converted = pd.to_numeric(
                original,
                errors="coerce"
            )

            non_null_original = original.notna().sum()
            non_null_converted = converted.notna().sum()

            if (
                non_null_original == 0
                or non_null_converted == non_null_original
            ):

                df[column] = converted

    # --------------------------------------------------------
    # Display cleaned data
    # --------------------------------------------------------

    print("\nCleaned borrower data:")

    for column in features:

        value = df.iloc[0][column]

        print(
            f"{column:30} : {value}"
        )

    print("\nCleaned data types:")

    for column in features:

        print(
            f"{column:30} : "
            f"{df[column].dtype}"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = df.isna().sum()

    missing_columns = [
        column
        for column in features
        if missing[column] > 0
    ]

    if missing_columns:

        print("\nMissing values detected:")

        for column in missing_columns:

            print(
                f"  - {column}: "
                f"{missing[column]}"
            )

        print(
            "\nThe trained model's imputer will "
            "handle these missing values."
        )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MICRO-LENDING BORROWER RISK ASSESSMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nTrained model not found.\n\n"
            "Run this first:\n"
            "python -m src.train_model"
        )

    print("\nLoading trained model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded successfully:",
        type(model).__name__
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = load_metadata()

    print("\nModel:")
    print(
        metadata.get(
            "best_model",
            "Unknown"
        )
    )

    print(
        "\nTraining records:",
        metadata.get(
            "training_rows",
            "Unknown"
        )
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = metadata.get(
        "features",
        []
    )

    if not features:

        raise RuntimeError(
            "No model features found in "
            "model_metadata.json."
        )

    print("\nFeatures used by model:")

    for feature in features:

        print(
            "  -",
            feature
        )

    # --------------------------------------------------------
    # Load MySQL data
    # --------------------------------------------------------

    print(
        "\nLoading borrower data from MySQL..."
    )

    try:

        df = load_loans(
            columns=features
        )

    except Exception as e:

        raise RuntimeError(
            "\nCould not load borrower data "
            "from MySQL.\n"
            f"Reason: {type(e).__name__}: {e}"
        )

    if df is None:

        raise RuntimeError(
            "load_loans() returned None."
        )

    if df.empty:

        raise RuntimeError(
            "No records were returned from "
            "loan_database.loan_sample."
        )

    print(
        f"Borrower records available: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    missing_columns = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "\nMissing model features in "
            "database result:\n\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_columns
            )
        )

    # --------------------------------------------------------
    # Select first borrower
    # --------------------------------------------------------

    borrower = df.iloc[[0]].copy()

    # --------------------------------------------------------
    # Clean borrower
    # --------------------------------------------------------

    borrower_df = clean_borrower_data(
        borrower,
        features
    )

    # --------------------------------------------------------
    # Create borrower dictionary
    # --------------------------------------------------------

    borrower_data = {}

    for feature in features:

        value = borrower_df.iloc[0][feature]

        if pd.isna(value):

            borrower_data[feature] = None

        else:

            borrower_data[feature] = value

    # --------------------------------------------------------
    # Display borrower
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BORROWER DATA USED FOR ASSESSMENT")
    print("=" * 70)

    for feature, value in borrower_data.items():

        print(
            f"{feature:30} : {value}"
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RUNNING MODEL PREDICTION")
    print("=" * 70)

    try:

        probabilities = model.predict_proba(
            borrower_df[features]
        )

    except Exception as e:

        print(
            "\nMODEL PREDICTION FAILED"
        )

        print(
            "=" * 70
        )

        print(
            f"Error type : {type(e).__name__}"
        )

        print(
            f"Error      : {e}"
        )

        print(
            "=" * 70
        )

        print(
            "\nData sent to model:"
        )

        print(
            borrower_df[
                features
            ].to_string(
                index=False
            )
        )

        raise

    # --------------------------------------------------------
    # Validate prediction
    # --------------------------------------------------------

    if len(probabilities) == 0:

        raise RuntimeError(
            "Model returned an empty prediction."
        )

    if len(probabilities[0]) < 2:

        raise RuntimeError(
            "Model did not return two class probabilities."
        )

    probability = float(
        probabilities[0][1]
    )

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    risk_score = round(
        (1 - probability) * 100,
        2
    )

    # --------------------------------------------------------
    # Risk category
    # --------------------------------------------------------

    risk_category = get_risk_category(risk_score)

    # --------------------------------------------------------
    # Assessment
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BORROWER ASSESSMENT")
    print("=" * 70)

    print(
        f"\nPredicted default probability : "
        f"{probability * 100:.2f}%"
    )

    print(
        f"Risk score                   : "
        f"{risk_score:.2f}/100"
    )

    print(
        f"Risk category                : "
        f"{risk_category}"
    )

    # --------------------------------------------------------
    # Eligibility / Decision
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ELIGIBILITY / DECISION")
    print("=" * 70)

    try:

        result = assess_borrower(
            borrower_data,
            probability,
            risk_score
        )

        if isinstance(
            result,
            dict
        ):

            for key, value in result.items():

                print(
                    f"{key:30} : {value}"
                )

        else:

            print(result)

    except Exception as e:

        print(
            "\nEligibility/decision engine "
            "could not be executed with the "
            "current database schema."
        )

        print(
            "Model prediction itself "
            "completed successfully."
        )

        print(
            f"Reason: {type(e).__name__}: {e}"
        )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ASSESSMENT COMPLETED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()