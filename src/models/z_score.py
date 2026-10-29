from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

file_path = Path(__file__).parent / "data" / "processed" / "creditcard_clean.csv"
df = pd.read_csv(file_path)

amount = df["Amount"]

#Среднее значение и стандартное отклонение
mean = amount.mean()
std = amount.std()

z_score = (amount - mean) / std

#Порог
threshold = 3

anomalies = df[np.abs(z_score) > threshold]

print("Средняя сумма транзакции: ", mean)
print("Стандартное отклонение: ", std)

print("\nКоличество найденных аномалий: ")
print(len(anomalies))

print("\nПервые найденные аномалии: ")
print(anomalies[["Time", "Amount", "Class"]].head(10))

print("\nИз найденных аномалий: ")
print("Мошеннических: ", (anomalies["Class"] == 1).sum())
print("Обычных: ", (anomalies["Class"] == 0).sum())

#Создаем предсказания алгоритма
y_pred = (np.abs(z_score) > threshold).astype(int)

#Считаем метрики
precision = precision_score(df["Class"], y_pred)
recall = recall_score(df["Class"], y_pred)
f1 = f1_score(df["Class"], y_pred)

print("\nОценка Z-score: ")
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)