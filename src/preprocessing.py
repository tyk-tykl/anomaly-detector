import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_features(df):
    """Простой вариант: все числовые столбцы, кроме Class (используется в run_experiment.py)."""
    X = df.drop(columns=["Class"], errors="ignore").select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("В файле нет числовых столбцов для анализа.")
    return X


def _to_datetime(s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce")


def _looks_like_dates(s):
    sample = s.dropna().astype(str).head(200)
    if sample.empty:
        return False
    return _to_datetime(sample).notna().mean() > 0.9


def build_features(df, feature_cols=None, label_col=None):
    """Превращает выбранные столбцы в числовую таблицу признаков:
    числа остаются как есть, даты дают час и день недели,
    текстовые столбцы кодируются частотой встречаемости."""
    cols = list(feature_cols) if feature_cols is not None else list(df.columns)
    cols = [c for c in cols if c != label_col]

    parts = {}
    for c in cols:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            parts[c] = s.astype(int)
        elif pd.api.types.is_numeric_dtype(s):
            parts[c] = s
        elif pd.api.types.is_datetime64_any_dtype(s) or _looks_like_dates(s):
            dt = _to_datetime(s)
            parts[f"{c}: час"] = dt.dt.hour
            parts[f"{c}: день недели"] = dt.dt.dayofweek
        else:
            parts[f"{c}: частота"] = s.map(s.value_counts(normalize=True))

    X = pd.DataFrame(parts, index=df.index)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("Не удалось получить ни одного признака из выбранных столбцов.")
    return X


def clean_and_scale(X_train, X_test):
    medians = X_train.median().fillna(0)
    X_train, X_test = X_train.fillna(medians), X_test.fillna(medians)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def add_features(X, mode="base"):
    """mode: base | log | log+hour"""
    X = X.copy()
    if mode in ("log", "log+hour") and "Amount" in X.columns:
        X["Amount"] = np.log1p(X["Amount"].clip(lower=0))
    if mode == "log+hour" and "Time" in X.columns:
        X["Time"] = (X["Time"] // 3600) % 24
    return X