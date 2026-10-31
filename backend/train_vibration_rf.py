
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "csv files"
OUTPUT_DIR = PROJECT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Change filenames here if your actual names differ.
FILES = {
    "Normal": "Normal_*.csv",
    "Imbalance": "Imbalance_*.csv",
    "Misalignment": "Misalignment_*.csv",
    "Looseness": "Looseness_*.csv",
}

FEATURES = [
    "rms_x", "peak_x", "kurtosis_x", "skewness_x",
    "dom_freq_x", "spectral_entropy_x",

    "rms_y", "peak_y", "kurtosis_y", "skewness_y",
    "dom_freq_y", "spectral_entropy_y",

    "rms_z", "peak_z", "kurtosis_z", "skewness_z",
    "dom_freq_z", "spectral_entropy_z",
]

TARGET = "label"
SESSION = "session_id"

RANDOM_STATE = 42

# ==========================================================
# 2. LOAD AND COMBINE THE FOUR CSV FILES
# ==========================================================

def load_dataset():
    frames = []

    for expected_label, pattern in FILES.items():
        paths = sorted(DATA_DIR.glob(pattern))

        if not paths:
            raise FileNotFoundError(
                f"No files found for {expected_label}: {pattern}\n"
                f"Folder: {DATA_DIR}"
            )

        print(f"\n{expected_label}: found {len(paths)} files")

        for path in paths:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip().str.lower()

            required = FEATURES + [SESSION]
            missing = [c for c in required if c not in df.columns]

            if missing:
                raise ValueError(
                    f"{path.name} is missing columns: {missing}"
                )

            # Use the known condition from the filename category
            df[TARGET] = expected_label

            # Give each recording its own unique group ID
            df["_group"] = (
                expected_label + "__" + path.stem
            )

            for col in FEATURES:
                df[col] = pd.to_numeric(
                    df[col], errors="coerce"
                )

            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=FEATURES + [SESSION])

            frames.append(df)

            print(f"  Loaded {path.name}: {len(df)} valid rows")

    if not frames:
        raise ValueError("No CSV data was loaded.")

    return pd.concat(frames, ignore_index=True)


# ==========================================================
# 3. CHECK CLASS BALANCE AND SESSION COUNTS
# ==========================================================

df = load_dataset()

print("\n========== DATASET SUMMARY ==========")
print("Total rows:", len(df))
print("Total sessions:", df["_group"].nunique())

print("\nRows per fault class:")
print(df[TARGET].value_counts().reindex(FILES.keys(), fill_value=0))

print("\nSessions per fault class:")
session_counts = df.groupby(TARGET)["_group"].nunique()
print(session_counts.reindex(FILES.keys(), fill_value=0))

df[TARGET].value_counts().reindex(
    FILES.keys(), fill_value=0
).rename_axis("class").to_csv(
    OUTPUT_DIR / "class_balance.csv"
)

session_counts.reindex(
    FILES.keys(), fill_value=0
).rename_axis("class").to_csv(
    OUTPUT_DIR / "session_counts.csv"
)

expected_classes = set(FILES.keys())
actual_classes = set(df[TARGET].unique())

if actual_classes != expected_classes:
    raise ValueError(
        f"Expected classes {expected_classes}, found {actual_classes}"
    )

# A meaningful holdout plus grouped CV needs multiple independent
# sessions per class. Three is the minimum for this workflow.
min_sessions = int(session_counts.min())

if min_sessions < 3:
    raise ValueError(
        "\nINSUFFICIENT INDEPENDENT SESSIONS.\n"
        f"Fewest sessions for any class: {min_sessions}.\n"
        "Collect at least 3 separate recording sessions per class "
        "for this script's grouped test and CV workflow. "
        "More sessions are preferable. Do not split rows from the "
        "same recording randomly to bypass this check."
    )

# ==========================================================
# 4. SESSION-BASED TRAIN/TEST SPLIT
# ==========================================================

X = df[FEATURES]
y = df[TARGET]
groups = df["_group"]

n_splits = min(5, min_sessions)

splitter = StratifiedGroupKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=RANDOM_STATE
)

all_classes = set(FILES.keys())
chosen_split = None

# Find a fold where both train and test contain all four classes.
for train_idx, test_idx in splitter.split(X, y, groups):
    if (
        set(y.iloc[train_idx]) == all_classes
        and set(y.iloc[test_idx]) == all_classes
    ):
        chosen_split = (train_idx, test_idx)
        break

if chosen_split is None:
    raise ValueError(
        "Could not create a grouped split containing all four "
        "classes in both train and test. Collect more sessions "
        "or inspect the class/session distribution."
    )

train_idx, test_idx = chosen_split

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

assert train_groups.isdisjoint(test_groups)

print("\n========== SESSION SPLIT ==========")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training sessions:", len(train_groups))
print("Testing sessions:", len(test_groups))
print("Training class counts:\n", y_train.value_counts())
print("Testing class counts:\n", y_test.value_counts())

# ==========================================================
# 5. GROUPED CROSS-VALIDATION ON TRAINING DATA ONLY
# ==========================================================

# Use fewer folds when there are fewer training sessions.
train_session_counts = (
    df.iloc[train_idx].groupby(TARGET)["_group"].nunique()
)

cv_folds = min(5, int(train_session_counts.min()))

if cv_folds < 2:
    raise ValueError(
        "Not enough training sessions per class for grouped CV."
    )

cv = StratifiedGroupKFold(
    n_splits=cv_folds,
    shuffle=True,
    random_state=RANDOM_STATE
)

rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

print(f"\n========== {cv_folds}-FOLD GROUPED CV ==========")
print("Searching for the best Random Forest settings...")

search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1,
    error_score="raise"
)

search.fit(X_train, y_train, groups=groups_train)

model = search.best_estimator_

print("\nBest parameters:")
print(search.best_params_)

print("Best grouped CV macro F1:",
      round(search.best_score_, 4))

# ==========================================================
# 6. EVALUATE ON UNSEEN TEST SESSIONS
# ==========================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    labels=list(FILES.keys()),
    output_dict=True,
    zero_division=0
)

print("\n========== FINAL TEST RESULTS ==========")
print(f"Test accuracy: {accuracy * 100:.2f}%")

print("\nPrecision / Recall / F1:")
print(classification_report(
    y_test,
    y_pred,
    labels=list(FILES.keys()),
    digits=4,
    zero_division=0
))

# Save report and predictions.
with open(OUTPUT_DIR / "classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

pd.DataFrame({
    "actual": y_test.to_numpy(),
    "predicted": y_pred,
    "session_id": df.iloc[test_idx][SESSION].to_numpy()
}).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

with open(OUTPUT_DIR / "model_summary.json", "w") as f:
    json.dump({
        "test_accuracy": accuracy,
        "grouped_cv_macro_f1": search.best_score_,
        "best_parameters": search.best_params_,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_sessions": len(train_groups),
        "test_sessions": len(test_groups),
        "features": FEATURES,
        "classes": list(FILES.keys())
    }, f, indent=2)

# ==========================================================
# 7. CONFUSION MATRIX
# ==========================================================

labels = list(FILES.keys())

cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(8, 6))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("Random Forest - Unseen Session Test")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
plt.close()

# ==========================================================
# 8. FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(importance["feature"], importance["importance"])
ax.invert_yaxis()
ax.set_xlabel("Feature importance")
ax.set_title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)
plt.close()

# ==========================================================
# 9. SAVE THE MODEL AND FEATURE ORDER
# ==========================================================

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "classes": labels
    },
    OUTPUT_DIR / "vibration_random_forest.pkl"
)

print("\n========== COMPLETE ==========")
print("Results folder:", OUTPUT_DIR)
print("Saved:")
for filename in [
    "class_balance.csv",
    "session_counts.csv",
    "classification_report.json",
    "test_predictions.csv",
    "model_summary.json",
    "confusion_matrix.png",
    "feature_importance.csv",
    "feature_importance.png",
    "vibration_random_forest.pkl",
]:
    print(" -", filename)

print("\nUse the TEST accuracy and per-class F1 scores "
      "to assess performance. Do not report training accuracy "
      "as test accuracy.")
