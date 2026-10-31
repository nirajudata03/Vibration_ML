"""
Predict motor fault classes from vibration feature CSV files.

Expected model file:
    results/vibration_random_forest.pkl

The model file should contain a dictionary with:
    model    -> trained classifier
    features -> list of feature column names used during training
    classes  -> class names

Example from Windows PowerShell or VS Code terminal:
    python predict_vibration_fault.py new_measurements.csv

With probabilities:
    python predict_vibration_fault.py new_measurements.csv --probabilities

Choose output file:
    python predict_vibration_fault.py new_measurements.csv --output predictions.csv

If your CSV has a few blank values and you want the script to fill them:
    python predict_vibration_fault.py new_measurements.csv --fill-missing median
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_FEATURES = [
    "rms_x",
    "peak_x",
    "kurtosis_x",
    "skewness_x",
    "dom_freq_x",
    "spectral_entropy_x",
    "rms_y",
    "peak_y",
    "kurtosis_y",
    "skewness_y",
    "dom_freq_y",
    "spectral_entropy_y",
    "rms_z",
    "peak_z",
    "kurtosis_z",
    "skewness_z",
    "dom_freq_z",
    "spectral_entropy_z",
]


def load_pickle_or_joblib(path: Path) -> Any:
    """Load a model file saved with either pickle or joblib."""
    try:
        import joblib  # type: ignore

        return joblib.load(path)
    except Exception:
        with path.open("rb") as file:
            return pickle.load(file)


def load_model_bundle(path: Path) -> tuple[Any, list[str], list[Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Train and save the model first, or pass the correct path with --model."
        )

    bundle = load_pickle_or_joblib(path)
    if not isinstance(bundle, dict):
        raise ValueError(
            "The model file must contain a dictionary with keys: model, features, classes."
        )

    missing_keys = {"model", "features", "classes"} - set(bundle)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"The model dictionary is missing required key(s): {missing}")

    model = bundle["model"]
    features = list(bundle["features"] or DEFAULT_FEATURES)
    classes = list(bundle["classes"])

    if not hasattr(model, "predict"):
        raise ValueError("The loaded model does not have a predict() method.")
    if not features:
        raise ValueError("The loaded model did not provide any feature names.")

    return model, features, classes


def validate_and_prepare_features(
    data: pd.DataFrame,
    features: list[str],
    fill_missing: str,
) -> pd.DataFrame:
    missing_columns = [column for column in features if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required feature column(s): "
            + ", ".join(missing_columns)
            + "\nMake sure the CSV contains the same feature columns used for training."
        )

    feature_data = data[features].copy()

    for column in features:
        feature_data[column] = pd.to_numeric(feature_data[column], errors="coerce")

    blank_counts = feature_data.isna().sum()
    columns_with_blanks = blank_counts[blank_counts > 0]

    if not columns_with_blanks.empty:
        details = ", ".join(
            f"{column}={count}" for column, count in columns_with_blanks.items()
        )

        if fill_missing == "error":
            raise ValueError(
                "Input CSV contains missing or non-numeric feature values: "
                + details
                + "\nFix the CSV, or run again with --fill-missing zero or --fill-missing median."
            )

        if fill_missing == "zero":
            feature_data = feature_data.fillna(0)
        elif fill_missing == "median":
            medians = feature_data.median(numeric_only=True)
            feature_data = feature_data.fillna(medians).fillna(0)
        else:
            raise ValueError(f"Unknown missing-data option: {fill_missing}")

        print(f"Filled missing/non-numeric values using {fill_missing}: {details}")

    return feature_data


def probability_column_name(class_name: Any) -> str:
    clean_name = str(class_name).strip().replace(" ", "_")
    return f"probability_{clean_name}"


def predict_file(
    model_path: Path,
    input_path: Path,
    output_path: Path | None,
    include_probabilities: bool,
    fill_missing: str,
) -> Path:
    model, features, classes = load_model_bundle(model_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    data = pd.read_csv(input_path)
    if data.empty:
        raise ValueError("Input CSV is empty. Add at least one row of feature values.")

    extra_columns = [column for column in data.columns if column not in features]
    if extra_columns:
        print(
            "Ignoring extra column(s) not used by the model: "
            + ", ".join(extra_columns)
        )

    feature_data = validate_and_prepare_features(data, features, fill_missing)
    predictions = model.predict(feature_data)

    result = data.copy()
    result["predicted_fault"] = predictions

    if include_probabilities:
        if not hasattr(model, "predict_proba"):
            print("Warning: this model does not support class probabilities.")
        else:
            probabilities = model.predict_proba(feature_data)
            probability_classes = list(getattr(model, "classes_", classes))

            for index, class_name in enumerate(probability_classes):
                result[probability_column_name(class_name)] = probabilities[:, index]

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_predictions.csv")

    result.to_csv(output_path, index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict motor fault classes from a vibration feature CSV file."
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to the new CSV file containing the 18 vibration feature columns.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("results") / "vibration_random_forest.pkl",
        help="Path to the saved model dictionary. Default: results/vibration_random_forest.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the output CSV. Default: input filename plus _predictions.csv",
    )
    parser.add_argument(
        "--probabilities",
        action="store_true",
        help="Add one probability column per class when the model supports predict_proba().",
    )
    parser.add_argument(
        "--fill-missing",
        choices=["error", "zero", "median"],
        default="error",
        help="How to handle missing or non-numeric feature values. Default: error",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        output_path = predict_file(
            model_path=args.model,
            input_path=args.csv_file,
            output_path=args.output,
            include_probabilities=args.probabilities,
            fill_missing=args.fill_missing,
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Saved predictions to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
