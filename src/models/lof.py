from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score

file_path = Path(__file__).parent / "data" / "processed" / "creditcard_clean.csv"
df = pd.read_csv(file_path)

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Создаем модель LOF
model = LocalOutlierFactor(
    n_neighbors=20,
    contamination="auto",
)

predictions = model.fit_predict(X_scaled)

y_pred = (predictions == -1).astype(int)

print("Количество найденных аномалий:")
print(y_pred.sum())

print("\nКоличество обычных транзакций: ")
print((y_pred == 0).sum())

print("\nКоличество всех транзакций: ")
print(len(y_pred))

#Считаем метрики
precision = precision_score(df["Class"], y_pred)
recall = recall_score(df["Class"], y_pred)
f1 = f1_score(df["Class"], y_pred)

print("\nОценка LOF: ")
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)