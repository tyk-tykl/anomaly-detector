import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import SEEDS, _top_k_metrics, get_labels
from src.models import fit_predict_score
from src.preprocessing import add_features, clean_and_scale, get_features

df = pd.read_csv(ROOT / "data" / "processed" / "creditcard_clean.csv")
y = get_labels(df)
X_full = get_features(df)

EXPERIMENTS = [
    # (набор признаков, алгоритм)
    ("base", "Z-score"),
    ("base", "Z-score (Amount)"),
    ("base", "Z-score (робастный)"),
    ("log", "Z-score"),
    ("log+hour", "Z-score"),
    ("base", "Isolation Forest"),
    ("log", "Isolation Forest"),
    ("log+hour", "Isolation Forest"),
    ("base", "LOF"),
    ("log", "LOF"),
    ("log+hour", "LOF"),
    ("base", "One-Class SVM"),
    ("log", "One-Class SVM"),
    ("log+hour", "One-Class SVM"),
]

rows = []
for mode, method in EXPERIMENTS:
    runs = []
    for seed in SEEDS:
        X = add_features(X_full, mode)
        X_tr, X_te, _, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        train, test = clean_and_scale(X_tr, X_te)
        amount_idx = list(X.columns).index("Amount")
        _, score = fit_predict_score(method, train, test, seed, amount_idx)
        runs.append({
            "Avg Precision": average_precision_score(y_te, score),
            "ROC-AUC": roc_auc_score(y_te, score),
            **_top_k_metrics(y_te, score),
        })
    r = pd.DataFrame(runs)
    rows.append({
        "Признаки": mode, "Алгоритм": method,
        "Avg Precision": r["Avg Precision"].mean(),
        "Avg Precision ±": r["Avg Precision"].std(),
        "ROC-AUC": r["ROC-AUC"].mean(),
        "Recall@1%": r["Recall@1%"].mean(),
        "Recall@1% ±": r["Recall@1%"].std(),
    })
    print(f"{mode:9s} {method:22s} AP={rows[-1]['Avg Precision']:.3f} "
          f"Recall@1%={rows[-1]['Recall@1%']:.3f}")

out = pd.DataFrame(rows).round(4)
(ROOT / "results").mkdir(exist_ok=True)
out.to_csv(ROOT / "results" / "experiment_features.csv", index=False, encoding="utf-8-sig")
print("\nСохранено в results/experiment_features.csv")