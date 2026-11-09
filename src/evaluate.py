import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .models import METHODS, fit_predict, fit_predict_score
from .preprocessing import clean_and_scale, get_features

MAX_TEST = 100000
MAX_CURVE_POINTS = 500
SEEDS = [42, 43, 44, 45, 46]  # пять запусков с разным разбиением
TOP_FRAC = 0.01               # доля транзакций для Precision@K / Recall@K
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MEAN_COLS = ["Precision", "Recall", "F1-score", "Avg Precision",
             "ROC-AUC", "Precision@1%", "Recall@1%"]
STD_COLS = ["F1-score", "Avg Precision", "ROC-AUC", "Recall@1%"]


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


def _pr_curve(y_true, score):
    precision, recall, _ = precision_recall_curve(y_true, score)
    if len(precision) > MAX_CURVE_POINTS:
        idx = np.linspace(0, len(precision) - 1, MAX_CURVE_POINTS).astype(int)
        precision, recall = precision[idx], recall[idx]
    return pd.DataFrame({"Recall": recall, "Precision": precision})


def _top_k_metrics(y_true, score, frac=TOP_FRAC):
    y = np.asarray(y_true)
    k = max(1, int(len(y) * frac))
    top = np.argsort(score)[::-1][:k]
    hits = int(y[top].sum())
    return {
        "Precision@1%": hits / k,
        "Recall@1%": hits / max(1, int(y.sum())),
    }


def evaluate_holdout(df, method, seed=42):
    y = get_labels(df)
    if y is None:
        raise ValueError("Для оценки нужен столбец Class со значениями 0 и 1 (оба класса).")

    X = get_features(df)
    strat = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, _, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=strat
    )

    if len(X_te) > MAX_TEST:
        strat = y_te if y_te.value_counts().min() >= 2 else None
        X_te, _, y_te, _ = train_test_split(
            X_te, y_te, train_size=MAX_TEST, random_state=seed, stratify=strat
        )

    train, test = clean_and_scale(X_tr, X_te)
    pred, score = fit_predict_score(method, train, test, seed)

    metrics = {
        **calc_metrics(y_te, pred),
        "Avg Precision": average_precision_score(y_te, score),
        "ROC-AUC": roc_auc_score(y_te, score),
        **_top_k_metrics(y_te, score),
        "Проверочных записей": len(y_te),
        "Найдено аномалий": int(pred.sum()),
    }
    return metrics, _pr_curve(y_te, score)


def compare_methods(df, seeds=SEEDS):
    y = get_labels(df)
    rows, curves = [], []
    for m in METHODS:
        start = time.perf_counter()
        if y is not None:
            runs, curve = [], None
            for seed in seeds:
                metrics, c = evaluate_holdout(df, m, seed)
                runs.append(metrics)
                if curve is None:
                    curve = c  # PR-кривая берётся из первого запуска
            runs_df = pd.DataFrame(runs)
            row = {"Алгоритм": m, **runs_df[MEAN_COLS].mean().to_dict()}
            for c in STD_COLS:
                row[f"{c} ±"] = runs_df[c].std()
            row["Проверочных записей"] = int(runs_df["Проверочных записей"].iloc[0])
            row["Найдено аномалий"] = int(round(runs_df["Найдено аномалий"].mean()))
            row["Время (сек.)"] = round((time.perf_counter() - start) / len(seeds), 2)
            curves.append(curve.assign(Алгоритм=m))
        else:
            found = int(detect_anomalies(df, m)["Anomaly"].sum())
            row = {"Алгоритм": m, "Найдено аномалий": found}
            row["Время (сек.)"] = round(time.perf_counter() - start, 2)
        rows.append(row)

    table = pd.DataFrame(rows)
    curves_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    baseline = float(y.mean()) if y is not None else None
    return table, curves_df, baseline


def save_comparison(table):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "comparison.csv"
    table.round(4).to_csv(path, index=False, encoding="utf-8-sig")
    return path