import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

MAX_TRAIN = 5000

MODELS = {
    "Isolation Forest": lambda: IsolationForest(random_state=42),
    "LOF": lambda: LocalOutlierFactor(n_neighbors=20, novelty=True),
    "One-Class SVM": lambda: OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
}
METHODS = ["Z-score", *MODELS]


def _subsample(X, seed=42):
    if len(X) <= MAX_TRAIN:
        return X
    idx = np.random.RandomState(seed).choice(len(X), MAX_TRAIN, replace=False)
    return X[idx]


def fit_predict(method, train, test):
    """train и test уже масштабированы. Возвращает массив 0/1."""
    if method == "Z-score":
        # после StandardScaler z-score — это просто значение признака
        return (np.abs(test) > 3).any(axis=1).astype(int)

    model = MODELS[method]()
    model.fit(_subsample(train) if method != "Isolation Forest" else train)
    return (model.predict(test) == -1).astype(int)