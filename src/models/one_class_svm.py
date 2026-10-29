from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score

file_path = Path(__file__).parent / "data" / "processed" / "creditcard_clean.csv"
df = pd.read_csv(file_path)

# Выбираем случайную часть датасета
df_sample = df.sample(n=10000, random_state=42)

# Отделяем признаки от правильных ответов
X = df_sample.drop("Class", axis=1)
y = df_sample["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.05
)

print("Начинаем обучение One-Class SVM...")
model.fit(X_scaled)
print("Обучение завершено!")

predictions = model.predict(X_scaled)

y_pred = (predictions == -1).astype(int)

print("Количество найденных аномалий: ")
print(y_pred.sum())

print("\nКоличество обычных транзакций: ")
print((y_pred == 0).sum())

print("\nКоличество всех транзакций: ")
print(len(y_pred))

#Считаем метрики
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("\nОценка One-Class SVM: ")
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)