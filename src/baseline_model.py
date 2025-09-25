#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv("data/autism_data.csv")

# ---------- build target y *first* (robust to strings or already-numeric) ----------
y_raw = df["ASD_traits"]

if y_raw.dtype == object:
    # handle case/whitespace variations
    y = y_raw.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
else:
    # already numeric? coerce safely to 0/1
    y = pd.to_numeric(y_raw, errors="coerce")

# keep only rows with a valid target
mask = y.notna()
df = df.loc[mask].copy()
y = y.loc[mask].astype(int)

# ---------- now clean features ----------
# Convert generic Yes/No features to 1/0
for col in df.columns:
    if col == "ASD_traits":
        continue
    if df[col].dtype == object:
        vals = set(str(v).strip().lower() for v in df[col].dropna().unique())
        if vals <= {"yes", "no"}:
            df[col] = df[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})

# Encode selected categorical columns if present
for col in ["Sex", "Ethnicity", "Who_completed_the_test"]:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Feature matrix (drop id/label columns)
X = df.drop(columns=["CASE_NO_PATIENT'S", "ASD_traits"], errors="ignore")

# Optional: simple numeric imputation (in case any numeric NaNs remain)
for c in X.columns:
    if X[c].dtype.kind in "biufc":
        X[c] = X[c].fillna(X[c].median())

# Sanity checks
print("Target distribution (after cleaning):")
print(y.value_counts(dropna=False))
print("X shape:", X.shape)

# Train/test split (only stratify if both classes present)
do_stratify = y.nunique() == 2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if do_stratify else None
)

# Baseline Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n== Logistic Regression Results ==")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
