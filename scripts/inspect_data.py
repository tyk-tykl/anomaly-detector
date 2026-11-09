from pathlib import Path
import pandas as pd

file_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "creditcard.csv"

df = pd.read_csv(file_path)

print("Размер датасета:")
print(df.shape)

print("\nНазвания столбцов:")
print(df.columns.tolist())

print("\nПервые пять транзакций:")
print(df.head())

print("\nТипы данных:")
print(df.dtypes)

print("\nКоличество пропусков:")
print(df.isnull().sum())

print("Есть ли пропуски: ")
print(df.isnull().values.any())

print("\nКоличество дубликатов: ")
print(df.duplicated().sum())

print("\nКоличество операций каждого класса: ")
print(df["Class"].value_counts())

print("\nПроцент операций каждого класса: ")
print(df["Class"].value_counts(normalize=True) * 100)

print("\nСтатистика Amount: ")
print(df["Amount"].describe())

print("\nСуммы обычных операций: ")
print(df[df["Class"] == 0]["Amount"].describe())

print("\nСуммы мошеннических операций: ")
print(df[df["Class"] == 1]["Amount"].describe())

print("\nПримеры дубликатов")
print(df[df.duplicated(keep=False)].head(10))