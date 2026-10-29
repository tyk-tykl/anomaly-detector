from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

file_path = Path(__file__).parent / "data" / "processed" / "creditcard_clean.csv"
df = pd.read_csv(file_path)

#Отделяем признаки от правильных ответов
X = df.drop("Class", axis=1)
y = df["Class"]

#Масштабируем признаки
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Создаем модель Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)

#Обучаем модель
model.fit(X_scaled)

#Получаем предсказания
predictions = model.predict(X_scaled)

#Преобразуем предсказания: -1 - аномалия, 1 - обычная транзакция
y_pred = (predictions == -1).astype(int)

print("Количество найденных аномалий: ")
print(y_pred.sum())

print("\nКоличество обычных транзакций: ")
print((y_pred == 0).sum())

print("\nКоличество всех транзакций: ")
print(len(y_pred))

df["Prediction"] = y_pred

anomalies = df[df["Prediction"] == 1]

print("\nПервые найденные аномалии: ")
print(anomalies[["Time", "Amount", "Class", "Prediction"]].head(10))

print("\nИз найденных аномалий: ")
print("Мошеннических: ", (anomalies["Class"] == 1).sum())
print("Обычных: ", (anomalies["Class"] == 0).sum())

#Считаем метрики
precision = precision_score(df["Class"], y_pred)
recall = recall_score(df["Class"], y_pred)
f1 = f1_score(df["Class"], y_pred)

print("\nОценка Isolation Forest: ")
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)