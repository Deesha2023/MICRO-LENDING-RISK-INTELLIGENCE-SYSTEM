# =========================================================
# RISK ENGINE
# =========================================================

def get_risk_category(risk_score):
    """Convert a 0-100 risk score into Low/Medium/High risk."""
    if risk_score >= 75:
        return "Low Risk"
    if risk_score >= 55:
        return "Medium Risk"
    return "High Risk"


def get_lending_decision(risk_category):
    """
    Project decision policy:
      Low Risk    -> Approve
      Medium Risk -> Forward to Manager / Manual Decision
      High Risk  -> Approve

    Note: approving high-risk borrowers follows the requested
    project rule. In a real lending system this policy should be
    validated by the lending institution.
    """
    if risk_category == "Medium Risk":
        return "Forward to Manager / Manual Decision", True
    return "Approve", True


def assess_borrower(borrower, probability, risk_score):
    reasons = []

    def num(key):
        try:
            value = borrower.get(key, 0)
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    loan_amount = num("loan_amnt")
    annual_income = num("annual_inc")
    dti = num("dti")
    int_rate = num("int_rate")
    revol_util = num("revol_util")
    grade = str(borrower.get("grade", "") or "").upper()
    verification = str(borrower.get("verification_status", "") or "").lower()
    employment = str(borrower.get("emp_length", "") or "").lower()

    if dti > 40:
        reasons.append("High debt-to-income ratio.")
    elif dti > 30:
        reasons.append("Moderately high debt-to-income ratio.")

    if revol_util > 80:
        reasons.append("Very high revolving credit utilization.")
    elif revol_util > 60:
        reasons.append("High revolving credit utilization.")

    if int_rate >= 20:
        reasons.append("High interest rate indicates elevated credit risk.")

    if grade in {"E", "F", "G"}:
        reasons.append("Lower loan grade indicates higher credit risk.")

    if annual_income > 0 and loan_amount / annual_income > 0.50:
        reasons.append("Loan amount is high relative to annual income.")

    if verification and "not verified" in verification:
        reasons.append("Borrower income is not verified.")

    if employment in {"< 1 year", "1 year"}:
        reasons.append("Limited employment history.")

    risk_category = get_risk_category(risk_score)
    decision, eligible = get_lending_decision(risk_category)

    if not reasons:
        reasons.append("No major rule-based risk indicators were identified.")

    return {
        "decision": decision,
        "eligible": eligible,
        "risk_category": risk_category,
        "reasons": reasons,
        "decision_policy": {
            "Low Risk": "Approve",
            "Medium Risk": "Forward to Manager / Manual Decision",
            "High Risk": "Approve"
        }
    }
