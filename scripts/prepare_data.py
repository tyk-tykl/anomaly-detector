from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = Path(__file__).parent / "data" / "processed" / "creditcard_clean.csv"
df = pd.read_csv(file_path)

#Отделяем признаки от правильных ответов
X = df.drop("Class", axis=1)
y = df["Class"]

print("\nПризнаки для анализа: ")
print(X.columns.tolist())

print("\nМетка Class: ")
print(y.name)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nРазмер масштабированных данных: ")
print(X_scaled.shape)

print("\nПроверка пропусков: ")
print(df.isnull().sum().sum())

print("\nКоличество дубликатов: ")
print(df.duplicated().sum())

print("\nКоличество обычных транзакций: ")
print((y == 0).sum())

print("\nКоличество мошеннических транзакций: ")
print((y == 1).sum())

print("\nРазмер исходных данных: ")
print(X.shape)

print("\nРазмер масштабированных признаков: ")
print(X_scaled.shape)

print("\nКоличество меток: ")
print(len(y))

print("\nПроверка количества данных: ")
print(len(X) == len(X_scaled) == len(y))