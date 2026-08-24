from pathlib import Path
import json
import os
import warnings
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sqlalchemy import create_engine, inspect, text
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "default_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

load_dotenv(ROOT / ".env")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "loan_database")
MYSQL_TABLE = os.getenv("MYSQL_TABLE", "loan_sample")

CANDIDATE_NUMERIC = [
    "loan_amnt","funded_amnt","funded_amnt_inv","installment","annual_inc",
    "dti","delinq_2yrs","inq_last_6mths","open_acc","pub_rec","revol_bal",
    "revol_util","total_acc","collections_12_mths_ex_med","acc_now_delinq",
    "tot_coll_amt","tot_cur_bal","total_rev_hi_lim","bc_open_to_buy","bc_util",
    "chargeoff_within_12_mths","delinq_amnt","mo_sin_old_rev_tl_op",
    "mo_sin_old_il_acct","mths_since_last_delinq","mths_since_last_major_derog",
    "mths_since_recent_bc","mths_since_recent_inq","num_accts_ever_120_pd",
    "num_actv_bc_tl","num_actv_rev_tl","num_bc_sats","num_bc_tl","num_il_tl",
    "num_op_rev_tl","num_rev_accts","num_rev_tl_bal_gt_0","num_sats",
    "num_tl_30dpd","num_tl_90g_dpd_24m","num_tl_op_past_12m","pct_tl_nvr_dlq",
    "percent_bc_gt_75","pub_rec_bankruptcies","tax_liens"
]
CANDIDATE_CATEGORICAL = [
    "term","grade","sub_grade","home_ownership","verification_status",
    "purpose","addr_state","initial_list_status","application_type","emp_length"
]
FORBIDDEN_COLUMNS = {
    "loan_condition","loan_status","id","member_id","url","desc","recoveries",
    "collection_recovery_fee","total_pymnt","total_pymnt_inv","total_rec_prncp",
    "total_rec_int","total_rec_late_fee","out_prncp","out_prncp_inv",
    "last_pymnt_d","last_pymnt_amnt","next_pymnt_d","last_credit_pull_d"
}

def get_engine():
    password = MYSQL_PASSWORD.replace("@", "%40")
    return create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{password}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}",
        pool_pre_ping=True
    )

def get_database_columns(engine):
    return [c["name"] for c in inspect(engine).get_columns(MYSQL_TABLE)]

def load_training_data(engine, available_columns):
    target = "loan_condition"
    if target not in available_columns:
        raise RuntimeError(f"'loan_condition' not found in {MYSQL_DATABASE}.{MYSQL_TABLE}.")
    numeric = [c for c in CANDIDATE_NUMERIC if c in available_columns and c not in FORBIDDEN_COLUMNS]
    categorical = [c for c in CANDIDATE_CATEGORICAL if c in available_columns and c not in FORBIDDEN_COLUMNS]
    features = numeric + categorical
    if not features:
        raise RuntimeError("No usable predictor columns were found.")
    cols = ", ".join(f"`{c}`" for c in [target] + features)
    df = pd.read_sql(text(f"SELECT {cols} FROM `{MYSQL_TABLE}`"), engine)
    return df, numeric, categorical

def prepare_target(df):
    df = df.copy()
    s = df["loan_condition"].astype(str).str.strip().str.lower()
    df["target"] = s.map({"good loan": 0, "bad loan": 1})
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)
    return df

def clean_numeric_columns(df, features):
    df = df.copy()
    for c in features:
        if c in df:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace("%","",regex=False).str.replace(",","",regex=False).str.strip(),
                errors="coerce"
            )
    return df

def remove_constant_features(df, numeric, categorical):
    numeric = [c for c in numeric if df[c].nunique(dropna=True) > 1]
    categorical = [c for c in categorical if df[c].nunique(dropna=True) > 1]
    return numeric, categorical

def build_pipeline(model, numeric, categorical):
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("model", model)
    ])

def evaluate_model(name, pipeline, X_test, y_test):
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["Good Loan", "Bad Loan"]
    }
    print("\n" + "="*70)
    print(name)
    print("="*70)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k.upper():12}: {v:.4f}")
    print("\nConfusion Matrix [rows=Actual, columns=Predicted]")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, labels=[0,1],
                                target_names=["Good Loan","Bad Loan"], zero_division=0))
    return metrics

def save_confusion_matrix(name, cm):
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    csv_path = REPORT_DIR / f"confusion_matrix_{safe}.csv"
    png_path = REPORT_DIR / f"confusion_matrix_{safe}.png"
    pd.DataFrame(cm, index=["Actual Good Loan","Actual Bad Loan"],
                 columns=["Predicted Good Loan","Predicted Bad Loan"]).to_csv(csv_path)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(cm)
        ax.set_xticks([0,1], ["Good Loan","Bad Loan"])
        ax.set_yticks([0,1], ["Good Loan","Bad Loan"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} - Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i,j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        png_path = None
    return str(csv_path), (str(png_path) if png_path else None)

def main():
    print("\n" + "="*70)
    print("MICRO-LENDING DEFAULT PREDICTION MODEL")
    print("Logistic Regression + Random Forest + Decision Tree")
    print("="*70)
    engine = get_engine()
    available_columns = get_database_columns(engine)
    df, numeric_features, categorical_features = load_training_data(engine, available_columns)
    df = prepare_target(df)
    df = clean_numeric_columns(df, numeric_features)
    numeric_features, categorical_features = remove_constant_features(df, numeric_features, categorical_features)
    features = numeric_features + categorical_features
    X, y = df[features], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced", max_depth=12
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced", max_depth=12, min_samples_leaf=5
        )
    }
    results, trained_models, confusion_files = {}, {}, {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = build_pipeline(model, numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results[name] = metrics
        trained_models[name] = pipeline
        confusion_files[name] = save_confusion_matrix(name, np.array(metrics["confusion_matrix"]))
        safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
        joblib.dump(pipeline, MODEL_DIR / f"{safe}.joblib")

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    joblib.dump(trained_models[best_name], MODEL_PATH)

    metadata = {
        "database": MYSQL_DATABASE, "table": MYSQL_TABLE,
        "training_rows": int(len(df)), "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)), "target_column": "loan_condition",
        "target_mapping": {"good loan": 0, "bad loan": 1},
        "numeric_features": numeric_features, "categorical_features": categorical_features,
        "features": features, "best_model": best_name, "model_metrics": results,
        "models": {name: f"models/{re.sub(r'[^A-Za-z0-9]+','_',name).strip('_').lower()}.joblib" for name in models},
        "confusion_matrix_files": confusion_files,
        "decision_policy": {
            "low_risk": "Approve",
            "medium_risk": "Forward to Manager / Manual Decision",
            "high_risk": "Approve"
        },
        "risk_thresholds": {"low_min_score": 75, "medium_min_score": 55}
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nBest model:", best_name)
    print("Saved:", MODEL_PATH)
    print("Confusion matrices saved under:", REPORT_DIR)
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
