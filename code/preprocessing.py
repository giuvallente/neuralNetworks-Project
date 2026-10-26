"""Leakage-safe feature preprocessing for the layoff survival classifier.

The project compares two feature sets proposed by the dataset author and
evaluates them with the same protocol: a stratified 80/20 development/test
split, followed by stratified 5-fold cross-validation inside the 80%.

Usage from a notebook in the project root::

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path("code").resolve()))
    from preprocessing import FEATURE_SETS, iter_cv_folds, fit_transform_dev_test

    # Cross-validation: the pipeline is re-fitted on the 4 training folds
    # of every iteration, so the validation fold never leaks into it.
    for fold in iter_cv_folds(X_dev, y_dev, FEATURE_SETS["extended"]):
        model.fit(fold.X_train, fold.y_train)
        ...evaluate on fold.X_val, fold.y_val...

    # Final model: fit on the whole 80%, transform the untouched 20% test set.
    pipeline, X_dev_ready, X_test_ready = fit_transform_dev_test(
        X_dev, X_test, FEATURE_SETS["extended"]
    )

The target (``laid_off``) is never used when fitting or transforming features.
"""

from typing import Iterator, NamedTuple, Optional, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
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
ALL_FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + BINARY_COLUMNS
FEATURE_COLUMNS = ALL_FEATURE_COLUMNS

# Feature sets suggested by the dataset author ("Modeling note").
# Both sets share every remaining predictor (salary_k_usd, department,
# mgmt_layer, level, is_recent_hire, high_salary, remote, ...); they differ
# only in the three structural drivers.
#   baseline : all predictors EXCEPT the structural drivers (performance_rating,
#              tenure_years and on_pip are included) -> expected to barely beat chance.
#   extended : baseline + team_wind_down, cost_center_type, role_redundant
#              -> expected strong.
STRUCTURAL_FEATURES = ["team_wind_down", "cost_center_type", "role_redundant"]
BASELINE_FEATURES = [c for c in ALL_FEATURE_COLUMNS if c not in STRUCTURAL_FEATURES]
FEATURE_SETS = {
    "baseline": BASELINE_FEATURES,
    "extended": BASELINE_FEATURES + STRUCTURAL_FEATURES,
}

RANDOM_STATE = 42
N_SPLITS = 5


def _check_columns(frame, feature_columns, name):
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing feature columns: {missing}")
    forbidden = sorted({"employee_id", "laid_off"} & set(frame.columns))
    if forbidden:
        raise ValueError(
            f"{name} contains {forbidden}; separate the target and "
            "remove the identifier before preprocessing."
        )


def make_preprocessing_pipeline(feature_columns: Optional[Sequence[str]] = None):
    """Build an unfitted pipeline for dense neural-network inputs.

    ``feature_columns`` selects which columns enter the model (default: all 13).
    Each column is routed to the numeric / categorical / binary branch it
    belongs to; branches left empty by the selection are omitted.
    """
    selected = list(ALL_FEATURE_COLUMNS if feature_columns is None else feature_columns)
    unknown = sorted(set(selected) - set(ALL_FEATURE_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown feature columns: {unknown}")

    numeric_cols = [c for c in NUMERIC_COLUMNS if c in selected]
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in selected]
    binary_cols = [c for c in BINARY_COLUMNS if c in selected]

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols))
    if binary_cols:
        transformers.append(("binary", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]), binary_cols))

    columns = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("features", columns)])


def _fit_transform(X_fit, X_others, feature_columns):
    """Fit on ``X_fit`` only; transform it and every frame in ``X_others``."""
    pipeline = make_preprocessing_pipeline(feature_columns)
    fitted = np.asarray(pipeline.fit_transform(X_fit[list(feature_columns)]),
                        dtype=np.float32)
    others = [
        np.asarray(pipeline.transform(X[list(feature_columns)]), dtype=np.float32)
        for X in X_others
    ]
    for values in [fitted, *others]:
        if not np.isfinite(values).all():
            raise ValueError("Array contains NaN or infinite values after preprocessing")
    return pipeline, fitted, others


class Fold(NamedTuple):
    """One cross-validation iteration, already preprocessed."""
    number: int            # 1-based
    pipeline: Pipeline     # fitted on this fold's training part only
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray


def iter_cv_folds(X_dev, y_dev, feature_columns: Sequence[str],
                  n_splits: int = N_SPLITS,
                  random_state: int = RANDOM_STATE) -> Iterator[Fold]:
    """Yield stratified K-fold iterations over the development set.

    The preprocessing pipeline is fitted on the training part of each fold
    (K-1 folds) and applied to the held-out fold, so no statistics of the
    validation fold (means, medians, categories) leak into training.
    """
    _check_columns(X_dev, feature_columns, "Development set")
    y_array = np.asarray(y_dev)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for number, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_array), start=1):
        pipeline, X_tr, (X_va,) = _fit_transform(
            X_dev.iloc[train_idx], [X_dev.iloc[val_idx]], feature_columns
        )
        yield Fold(number, pipeline, X_tr, X_va, y_array[train_idx], y_array[val_idx])


def fit_transform_dev_test(X_dev, X_test, feature_columns: Sequence[str],
                           verbose: bool = True):
    """Fit on the whole development set (80%); transform the untouched test set.

    Use for the final model after cross-validation has chosen hyper-parameters.
    Returns ``(pipeline, dev_array, test_array)`` as ``float32`` arrays.
    """
    _check_columns(X_dev, feature_columns, "Development set")
    _check_columns(X_test, feature_columns, "Test set")
    pipeline, dev, (test,) = _fit_transform(X_dev, [X_test], feature_columns)

    if verbose:
        names = pipeline.get_feature_names_out()
        print(f"Development: shape={dev.shape}, NaN values={int(np.isnan(dev).sum())}")
        print(f"Test:        shape={test.shape}, NaN values={int(np.isnan(test).sum())}")
        print(f"Final feature count: {len(names)}")
        for feature in names:
            print(f"  {feature}")
    return pipeline, dev, test