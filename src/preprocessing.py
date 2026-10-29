import numpy as np
from sklearn.preprocessing import StandardScaler


def get_features(df):
    X = df.drop(columns=["Class"], errors="ignore").select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("В файле нет числовых столбцов для анализа.")
    return X


def clean_and_scale(X_train, X_test):
    medians = X_train.median().fillna(0)
    X_train, X_test = X_train.fillna(medians), X_test.fillna(medians)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)