import time

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .models import METHODS, fit_predict
from .preprocessing import clean_and_scale, get_features

MAX_TEST = 100000


def get_labels(df):
    if "Class" not in df.columns:
        return None
    y = pd.to_numeric(df["Class"], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all() or y.nunique() < 2:
        return None
    return y.astype(int)


def calc_metrics(y_true, y_pred):
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }


def detect_anomalies(df, method):
    X = get_features(df)
    train, test = clean_and_scale(X, X)
    result = df.drop(columns=["Class"], errors="ignore").copy()
    result["Anomaly"] = fit_predict(method, train, test)
    return result


def evaluate_holdout(df, method):
    y = get_labels(df)
    if y is None:
        raise ValueError("Для оценки нужен столбец Class со значениями 0 и 1 (оба класса).")

    X = get_features(df)
    strat = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, _, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    if len(X_te) > MAX_TEST:
        strat = y_te if y_te.value_counts().min() >= 2 else None
        X_te, _, y_te, _ = train_test_split(
            X_te, y_te, train_size=MAX_TEST, random_state=42, stratify=strat
        )

    train, test = clean_and_scale(X_tr, X_te)
    pred = fit_predict(method, train, test)

    return {
        **calc_metrics(y_te, pred),
        "Проверочных записей": len(y_te),
        "Найдено аномалий": int(pred.sum()),
    }


def compare_methods(df):
    has_labels = get_labels(df) is not None
    rows = []
    for m in METHODS:
        start = time.perf_counter()
        if has_labels:
            q = evaluate_holdout(df, m)
            row = {"Алгоритм": m, **q}
        else:
            found = int(detect_anomalies(df, m)["Anomaly"].sum())
            row = {"Алгоритм": m, "Найдено аномалий": found}
        row["Время (сек.)"] = round(time.perf_counter() - start, 2)
        rows.append(row)
    return pd.DataFrame(rows)