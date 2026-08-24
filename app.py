import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import create_model_input
from src.risk_engine import assess_borrower


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent

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
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="Micro-Lending Risk Intelligence",
    page_icon="💳",
    layout="wide",
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}\n\n"
            "Run:\n"
            "python -m src.train_model"
        )

    model = joblib.load(
        MODEL_PATH
    )

    metadata = {}

    if METADATA_PATH.exists():

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(
                file
            )

    return model, metadata


# =========================================================
# LOAD
# =========================================================

try:

    model, metadata = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained model."
    )

    st.exception(e)

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.title(
    "💳 Micro-Lending Risk Intelligence Platform"
)

st.markdown(
    """
    ### AI-Powered Loan Default Risk Assessment

    This platform uses machine learning and
    rule-based risk analysis to evaluate borrower
    default risk.
    """
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🤖 Model Information"
)

st.sidebar.write(
    "**Model:** "
    + str(
        metadata.get(
            "best_model",
            "Unknown"
        )
    )
)

st.sidebar.write(
    "**Training Rows:** "
    + str(
        metadata.get(
            "training_rows",
            "Unknown"
        )
    )
)

st.sidebar.write(
    "**Database:** "
    + str(
        metadata.get(
            "database",
            "loan_database"
        )
    )
)

st.sidebar.write(
    "**Table:** "
    + str(
        metadata.get(
            "table",
            "loan_sample"
        )
    )
)


# =========================================================
# INPUT
# =========================================================

st.header(
    "👤 Borrower Information"
)

with st.form(
    "borrower_form"
):

    col1, col2, col3 = st.columns(3)

    # -----------------------------------------------------
    # Financial
    # -----------------------------------------------------

    with col1:

        st.subheader(
            "Financial Information"
        )

        loan_amnt = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

        annual_inc = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=50000.0,
            step=5000.0
        )

        int_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=12.0,
            step=0.1
        )

        dti = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0,
            max_value=200.0,
            value=15.0,
            step=0.5
        )

        revol_util = st.number_input(
            "Revolving Utilization (%)",
            min_value=0.0,
            max_value=200.0,
            value=30.0,
            step=1.0
        )

    # -----------------------------------------------------
    # Loan
    # -----------------------------------------------------

    with col2:

        st.subheader(
            "Loan Information"
        )

        term = st.selectbox(
            "Loan Term",
            [
                "36 months",
                "60 months"
            ]
        )

        grade = st.selectbox(
            "Loan Grade",
            [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G"
            ]
        )

        sub_grade = st.selectbox(
            "Sub Grade",
            [
                "A1", "A2", "A3", "A4", "A5",
                "B1", "B2", "B3", "B4", "B5",
                "C1", "C2", "C3", "C4", "C5",
                "D1", "D2", "D3", "D4", "D5",
                "E1", "E2", "E3", "E4", "E5",
                "F1", "F2", "F3", "F4", "F5",
                "G1", "G2", "G3", "G4", "G5"
            ]
        )

        purpose = st.selectbox(
            "Loan Purpose",
            [
                "debt_consolidation",
                "credit_card",
                "home_improvement",
                "major_purchase",
                "small_business",
                "car",
                "medical",
                "moving",
                "vacation",
                "wedding",
                "other"
            ]
        )

        application_type = st.selectbox(
            "Application Type",
            [
                "Individual",
                "Joint App"
            ]
        )

    # -----------------------------------------------------
    # Borrower
    # -----------------------------------------------------

    with col3:

        st.subheader(
            "Borrower Profile"
        )

        home_ownership = st.selectbox(
            "Home Ownership",
            [
                "RENT",
                "OWN",
                "MORTGAGE",
                "OTHER"
            ]
        )

        verification_status = st.selectbox(
            "Verification Status",
            [
                "Verified",
                "Source Verified",
                "Not Verified"
            ]
        )

        emp_length = st.selectbox(
            "Employment Length",
            [
                "< 1 year",
                "1 year",
                "2 years",
                "3 years",
                "4 years",
                "5 years",
                "6 years",
                "7 years",
                "8 years",
                "9 years",
                "10+ years"
            ]
        )

    submitted = st.form_submit_button(
        "🔍 Assess Borrower Risk",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if submitted:

    if loan_amnt <= 0:

        st.error(
            "Loan amount must be greater than zero."
        )

        st.stop()

    if annual_inc <= 0:

        st.error(
            "Annual income must be greater than zero."
        )

        st.stop()

    borrower = {

        "loan_amnt": loan_amnt,

        "annual_inc": annual_inc,

        "int_rate": int_rate,

        "dti": dti,

        "revol_util": revol_util,

        "term": term,

        "grade": grade,

        "sub_grade": sub_grade,

        "purpose": purpose,

        "application_type": application_type,

        "home_ownership": home_ownership,

        "verification_status": verification_status,

        "emp_length": emp_length,
    }

    try:

        # -------------------------------------------------
        # Prepare input
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
            assess_borrower(
                borrower,
                probability,
                risk_score
            )
            .get(
                "risk_category",
                "Unknown"
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
        # Results
        # -------------------------------------------------

        st.divider()

        st.header(
            "📊 Risk Assessment Result"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Default Probability",
                f"{probability * 100:.2f}%"
            )

        with c2:

            st.metric(
                "Risk Score",
                f"{risk_score:.2f}/100"
            )

        with c3:

            st.metric(
                "Risk Category",
                risk_category
            )

        with c4:

            st.metric(
                "Prediction",
                str(prediction)
            )

        # -------------------------------------------------
        # Decision
        # -------------------------------------------------

        st.subheader(
            "🏦 Lending Decision"
        )

        decision = assessment.get(
            "decision",
            "Review"
        )

        eligible = assessment.get(
            "eligible",
            False
        )

        if decision == "Approve":
            st.success("✅ LOAN APPROVED")
        elif decision == "Forward to Manager / Manual Decision":
            st.warning("⚠️ FORWARD TO MANAGER / MANUAL DECISION")
        else:
            st.error("❌ REJECT")

        st.write(
            "**Eligible:** "
            + (
                "Yes"
                if eligible
                else "No"
            )
        )

        # -------------------------------------------------
        # Reasons
        # -------------------------------------------------

        st.subheader(
            "🔎 Risk Factors"
        )

        reasons = assessment.get(
            "reasons",
            []
        )

        for reason in reasons:

            st.write(
                "• " + str(reason)
            )

        # -------------------------------------------------
        # Model Comparison + Confusion Matrices
        # -------------------------------------------------

        st.subheader("🤖 Model Comparison")

        model_metrics = metadata.get("model_metrics", {})
        if model_metrics:
            comparison_rows = []
            for model_name, values in model_metrics.items():
                if not isinstance(values, dict) or "accuracy" not in values:
                    comparison_rows.append({
                        "Model": model_name,
                        "Accuracy": "Not trained",
                        "Precision": "Not trained",
                        "Recall": "Not trained",
                        "F1 Score": "Not trained",
                        "ROC-AUC": "Not trained"
                    })
                    continue
                comparison_rows.append({
                    "Model": model_name,
                    "Accuracy": f"{values.get('accuracy', 0) * 100:.2f}%",
                    "Precision": f"{values.get('precision', 0) * 100:.2f}%",
                    "Recall": f"{values.get('recall', 0) * 100:.2f}%",
                    "F1 Score": f"{values.get('f1', 0) * 100:.2f}%",
                    "ROC-AUC": f"{values.get('roc_auc', 0) * 100:.2f}%"
                })
            st.dataframe(pd.DataFrame(comparison_rows),
                         use_container_width=True, hide_index=True)

        st.subheader("📊 Confusion Matrix")

        confusion_files = metadata.get("confusion_matrix_files", {})
        matrix_tabs = st.tabs(["Logistic Regression", "Random Forest", "Decision Tree"])
        for tab, model_name in zip(matrix_tabs,
                                   ["Logistic Regression", "Random Forest", "Decision Tree"]):
            with tab:
                values = model_metrics.get(model_name, {})
                cm = values.get("confusion_matrix")
                if cm:
                    cm_df = pd.DataFrame(
                        cm,
                        index=["Actual Good Loan", "Actual Bad Loan"],
                        columns=["Predicted Good Loan", "Predicted Bad Loan"]
                    )
                    st.dataframe(cm_df, use_container_width=True)
                image_info = confusion_files.get(model_name)
                if image_info and len(image_info) > 1 and image_info[1]:
                    image_path = Path(image_info[1])
                    if not image_path.is_absolute():
                        image_path = ROOT / image_path
                    if image_path.exists():
                        st.image(str(image_path), use_container_width=True)

        # -------------------------------------------------
        # Borrower Summary
        # -------------------------------------------------

        st.subheader(
            "👤 Borrower Summary"
        )

        summary = pd.DataFrame(
            {
                "Parameter": [
                    "Loan Amount",
                    "Annual Income",
                    "Interest Rate",
                    "DTI",
                    "Revolving Utilization",
                    "Term",
                    "Grade",
                    "Sub Grade",
                    "Purpose",
                    "Home Ownership",
                    "Employment Length",
                    "Verification Status",
                    "Application Type"
                ],

                "Value": [
                    f"{loan_amnt:,.2f}",
                    f"{annual_inc:,.2f}",
                    f"{int_rate:.2f}%",
                    f"{dti:.2f}",
                    f"{revol_util:.2f}%",
                    term,
                    grade,
                    sub_grade,
                    purpose,
                    home_ownership,
                    emp_length,
                    verification_status,
                    application_type
                ]
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # Model Information
        # -------------------------------------------------

        with st.expander(
            "🤖 Model Details"
        ):

            st.write("**Best Model:**", metadata.get("best_model", "Unknown"))
            st.write("**Models Trained:**", ["Logistic Regression", "Random Forest", "Decision Tree"])
            st.write("**Decision Policy:**", metadata.get("decision_policy", {}))

            st.write(
                "**Features:**",
                metadata.get(
                    "features",
                    []
                )
            )

            st.write(
                "**Metrics:**",
                metadata.get(
                    "model_metrics",
                    {}
                )
            )

    except Exception as e:

        st.error(
            "Risk assessment failed."
        )

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Micro-Lending Risk Intelligence Platform | "
    "AI/ML Credit Risk Assessment"
)