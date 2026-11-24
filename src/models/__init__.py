import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

MAX_TRAIN = 5000
SVM_TRAIN = 2500

MODELS = {
    "Isolation Forest": lambda seed: IsolationForest(random_state=seed),
    "LOF": lambda seed: LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1),
    "One-Class SVM": lambda seed: OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
}
METHODS = ["Z-score", *MODELS]


def _subsample(X, size=MAX_TRAIN, seed=42):
    if len(X) <= size:
        return X
    idx = np.random.RandomState(seed).choice(len(X), size, replace=False)
    return X[idx]


def _parallel(func, X, chunk=20000):
    parts = np.array_split(X, max(1, len(X) // chunk))
    results = Parallel(n_jobs=-1)(delayed(func)(p) for p in parts)
    return np.concatenate(results)


def fit_predict_score(method, train, test, seed=42, amount_idx=None):
    if method == "Z-score":
        score = np.abs(test).max(axis=1)
        return (score > 3).astype(int), score

    if method == "Z-score (Amount)":
        if amount_idx is None:
            raise ValueError("Не найден столбец Amount.")
        score = np.abs(test[:, amount_idx])
        return (score > 3).astype(int), score

    if method == "Z-score (робастный)":
        med = np.median(train, axis=0)
        mad = np.median(np.abs(train - med), axis=0)
        mad = np.where(mad == 0, 1e-9, mad)
        score = (np.abs(test - med) / (1.4826 * mad)).max(axis=1)
        return (score > 3).astype(int), score

    if method not in MODELS:
        raise ValueError("Неизвестный алгоритм.")

    model = MODELS[method](seed)

    if method == "Isolation Forest":
        model.fit(train)
    elif method == "One-Class SVM":
        model.fit(_subsample(train, SVM_TRAIN, seed))
    else:
        model.fit(_subsample(train, MAX_TRAIN, seed))

    if method == "One-Class SVM":
        dec = _parallel(model.decision_function, test)
    else:
        dec = model.decision_function(test)

    return (dec < 0).astype(int), -dec


def fit_predict(method, train, test):
    return fit_predict_score(method, train, test)[0]