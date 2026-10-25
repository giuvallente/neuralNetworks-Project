"""Training-only feature preprocessing for the layoff survival classifier.

Usage from a notebook in the project root::

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path("code").resolve()))
    from preprocessing import fit_transform_splits

    preprocessor, X_train_ready, X_val_ready, X_test_ready = (
        fit_transform_splits(X_train, X_val, X_test)
    )

Keep ``y_train``, ``y_val``, and ``y_test`` separate; the target is not used
when fitting or transforming the features.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = ["tenure_years", "performance_rating", "salary_k_usd"]
CATEGORICAL_COLUMNS = [
    "department", "cost_center_type", "mgmt_layer", "level",
]
BINARY_COLUMNS = [
    "team_wind_down", "is_recent_hire", "on_pip",
    "high_salary", "remote", "role_redundant",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + BINARY_COLUMNS


def make_preprocessing_pipeline():
    """Build an unfitted pipeline for dense neural-network inputs."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binary = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    columns = ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_COLUMNS),
            ("categorical", categorical, CATEGORICAL_COLUMNS),
            ("binary", binary, BINARY_COLUMNS),
        ],
        remainder="drop",
    )
    return Pipeline([("features", columns)])


def fit_transform_splits(X_train, X_val, X_test):
    """Fit on train only; transform validation and test with the fitted pipeline.

    Returns ``(pipeline, train_array, val_array, test_array)``. All arrays
    have the same ordered columns and ``float32`` dtype for neural networks.
    """
    splits = {"Train": X_train, "Validation": X_val, "Test": X_test}
    for name, frame in splits.items():
        missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing feature columns: {missing}")
        forbidden = sorted({"employee_id", "laid_off"} & set(frame.columns))
        if forbidden:
            raise ValueError(
                f"{name} contains {forbidden}; separate the target and "
                "remove the identifier before preprocessing."
            )

    pipeline = make_preprocessing_pipeline()
    train = np.asarray(pipeline.fit_transform(X_train), dtype=np.float32)
    validation = np.asarray(pipeline.transform(X_val), dtype=np.float32)
    test = np.asarray(pipeline.transform(X_test), dtype=np.float32)
    feature_names = pipeline.get_feature_names_out()

    for name, values in zip(splits, (train, validation, test)):
        missing_count = int(np.isnan(values).sum())
        print(f"{name}: shape={values.shape}, NaN values={missing_count}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinite values after preprocessing")

    print(f"Final feature count: {len(feature_names)}")
    print("Feature names:")
    for feature in feature_names:
        print(f"  {feature}")

    return pipeline, train, validation, test
