import os
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

st.set_page_config(page_title="Autism Traits Predictor", page_icon="🧠", layout="wide")

st.title("🧠 Autism Traits Predictor")
st.write("Upload a CSV or use the sample to train **Logistic Regression** and **Random Forest**, then see metrics and plots.")

# ---------- data loading ----------
def load_default():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "data", "autism_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = load_default()
    if df is not None:
        st.info("Using sample dataset from `data/autism_data.csv`.")
    else:
        st.warning("No file uploaded and no sample found. Please upload a CSV to continue.")
        st.stop()

st.write("**Preview:**")
st.dataframe(df.head(10))

# ---------- preprocessing ----------
def build_target_and_features(df: pd.DataFrame):
    # Target first (robust to both string and numeric)
    y_col = df["ASD_traits"]
    if y_col.dtype == object:
        y = y_col.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    else:
        y = pd.to_numeric(y_col, errors="coerce")
    mask = y.notna()
    df = df.loc[mask].copy()
    y = y.loc[mask].astype(int)

    # Convert generic Yes/No columns to 1/0 (excluding the target)
    for col in df.columns:
        if col == "ASD_traits":
            continue
        if df[col].dtype == object:
            vals = set(str(v).strip().lower() for v in df[col].dropna().unique())
            if vals <= {"yes", "no"}:
                df[col] = df[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})

    # Encode selected categoricals if present
    for col in ["Sex", "Ethnicity", "Who_completed_the_test"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Feature matrix
    X = df.drop(columns=["CASE_NO_PATIENT'S", "ASD_traits"], errors="ignore")

    # Simple numeric imputation
    for c in X.columns:
        if X[c].dtype.kind in "biufc":
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    return X, y

try:
    X, y = build_target_and_features(df)
except KeyError as e:
    st.error(f"Missing expected column: {e}. Make sure your CSV has `ASD_traits`.")
    st.stop()

st.write(f"**Dataset after cleaning:** {X.shape[0]} rows, {X.shape[1]} features")
st.write("**Target distribution:**")
st.write(y.value_counts().rename({0: "No", 1: "Yes"}))

# ---------- train/test split ----------
if y.nunique() < 2:
    st.error("Your dataset has only one class in `ASD_traits` after cleaning. Add more diverse data.")
    st.stop()

test_size = st.sidebar.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5) / 100.0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# ---------- models ----------
# Logistic Regression (with scaling)
scaler = StandardScaler(with_mean=False)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
y_proba_lr = lr.predict_proba(X_test_s)[:, 1]

# Random Forest
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# ---------- metrics ----------
def show_metrics(name, y_true, y_pred, y_proba):
    st.subheader(f"{name} Metrics")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.json(report)

    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix – {name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    st.pyplot(fig)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {name}")
    plt.legend(loc="lower right")
    st.pyplot(fig)

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig = plt.figure()
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall – {name}")
    plt.legend(loc="lower left")
    st.pyplot(fig)

    return report, roc_auc, ap

col1, col2 = st.columns(2)
with col1:
    r_lr, auc_lr, ap_lr = show_metrics("Logistic Regression", y_test, y_pred_lr, y_proba_lr)
with col2:
    r_rf, auc_rf, ap_rf = show_metrics("Random Forest", y_test, y_pred_rf, y_proba_rf)

# ---------- feature importance ----------
st.subheader("Random Forest – Top Feature Importances")
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(15)
st.dataframe(feat_imp.to_frame("importance"))

fig = plt.figure()
feat_imp.iloc[::-1].plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top-15 Feature Importances (RF)")
st.pyplot(fig)

# ---------- threshold tuning ----------
st.subheader("Threshold Tuning (Random Forest)")
threshold = st.slider("Decision Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
y_thresh = (y_proba_rf >= threshold).astype(int)
cm_t = confusion_matrix(y_test, y_thresh)
fig = plt.figure()
plt.imshow(cm_t, interpolation='nearest')
plt.title(f"Confusion Matrix – RF @ threshold={threshold:.2f}")
plt.xlabel('Predicted')
plt.ylabel('True')
for (i, j), v in np.ndenumerate(cm_t):
    plt.text(j, i, str(v), ha='center', va='center')
st.pyplot(fig)

# ---------- footer ----------
st.markdown("---")
st.caption("Built with Streamlit • Models retrained on current dataset each run for demo purposes.")
