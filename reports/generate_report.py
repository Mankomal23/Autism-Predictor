#!/usr/bin/env python3
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, "data")
REPORTS_DIR = os.path.join(BASE, "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_and_prepare():
    df = pd.read_csv(os.path.join(DATA_DIR, "autism_data.csv"))

    # Target first (robust to already-encoded or string)
    y_col = df["ASD_traits"]
    if y_col.dtype == object:
        y = y_col.astype(str).str.strip().str.lower().map({"yes":1, "no":0})
    else:
        y = pd.to_numeric(y_col, errors="coerce")
    mask = y.notna()
    df = df.loc[mask].copy()
    y = y.loc[mask].astype(int)

    # Now convert Yes/No in remaining object columns
    for col in df.columns:
        if col == "ASD_traits":
            continue
        if df[col].dtype == object:
            uniq = set(str(v).strip().lower() for v in df[col].dropna().unique())
            if uniq <= {"yes", "no"}:
                df[col] = df[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})

    # Encode categoricals
    for col in ["Sex","Ethnicity","Who_completed_the_test"]:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Features
    X = df.drop(columns=["CASE_NO_PATIENT'S","ASD_traits"], errors="ignore")

    # Impute / encode leftovers
    for c in X.columns:
        if X[c].dtype.kind in "biufc":
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    return X, y
def plot_confusion(cm, title, path):
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression
    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    y_proba_lr = lr.predict_proba(X_test_s)[:,1]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:,1]

    # Metrics
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_proba_lr)
    prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_proba_rf)
    ap_lr = average_precision_score(y_test, y_proba_lr)
    ap_rf = average_precision_score(y_test, y_proba_rf)

    # Save plots
    # ROC
    plt.figure()
    plt.plot(fpr_lr, tpr_lr, label=f'LR AUC = {roc_auc_lr:.3f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC – Logistic Regression')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'roc_lr.png'))
    plt.close()

    plt.figure()
    plt.plot(fpr_rf, tpr_rf, label=f'RF AUC = {roc_auc_rf:.3f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC – Random Forest')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'roc_rf.png'))
    plt.close()

    # PR
    plt.figure()
    plt.plot(rec_lr, prec_lr, label=f'LR AP = {ap_lr:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall – Logistic Regression')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'pr_lr.png'))
    plt.close()

    plt.figure()
    plt.plot(rec_rf, prec_rf, label=f'RF AP = {ap_rf:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall – Random Forest')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'pr_rf.png'))
    plt.close()

    # Confusion matrices
    plot_confusion(cm_lr, "Confusion Matrix – LR", os.path.join(FIG_DIR, "cm_lr.png"))
    plot_confusion(cm_rf, "Confusion Matrix – RF", os.path.join(FIG_DIR, "cm_rf.png"))

    # Classification reports
    report_lr = classification_report(y_test, y_pred_lr, output_dict=True)
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

    metrics = {
        "lr": {
            "roc_auc": float(roc_auc_lr),
            "avg_precision": float(ap_lr),
            "report": report_lr
        },
        "rf": {
            "roc_auc": float(roc_auc_rf),
            "avg_precision": float(ap_rf),
            "report": report_rf
        }
    }

    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Build Markdown summary
    md = []
    md.append("# Autism Predictor – Summary")
    md.append("")
    md.append("## Models Compared")
    md.append("- Logistic Regression")
    md.append("- Random Forest")
    md.append("")
    md.append("## Key Metrics")
    md.append(f"- **LR**: ROC AUC = {roc_auc_lr:.3f}, Average Precision = {ap_lr:.3f}")
    md.append(f"- **RF**: ROC AUC = {roc_auc_rf:.3f}, Average Precision = {ap_rf:.3f}")
    md.append("")
    md.append("## Curves")
    md.append("![ROC LR](figures/roc_lr.png)")
    md.append("![ROC RF](figures/roc_rf.png)")
    md.append("")
    md.append("![PR LR](figures/pr_lr.png)")
    md.append("![PR RF](figures/pr_rf.png)")
    md.append("")
    md.append("## Confusion Matrices")
    md.append("![CM LR](figures/cm_lr.png)")
    md.append("![CM RF](figures/cm_rf.png)")
    md.append("")
    md.append("## Notes")
    md.append("- ROC AUC summarizes ranking quality; higher is better.")
    md.append("- Average Precision emphasizes performance on the positive class if classes are imbalanced.")
    md.append("- Use EDA insights to decide which features to engineer or drop.")

    with open(os.path.join(REPORTS_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\\n".join(md))

if __name__ == "__main__":
    main()
