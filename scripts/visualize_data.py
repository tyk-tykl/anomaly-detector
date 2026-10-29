from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

file_path = Path(__file__).parent / "data" / "raw" / "creditcard.csv"

df = pd.read_csv(file_path)

#Считаем кол-во операций каждого класса
class_counts = df["Class"].value_counts()

#Создаем график
plt.bar(["Обычные", "Мошеннические"], class_counts)

plt.title("Количество обычных и мошеннических транзакций")
plt.xlabel("Тип транзакции")
plt.ylabel("Количество")

#Выводим график
plt.show()

#График распредления сумм
plt.hist(df["Amount"], bins=50)

plt.title("Распределение сумм транзакций")
plt.xlabel("Сумма транзакции")
plt.ylabel("Количество транзакций")

plt.show()

#Распределение сумм обычных и мошеннических транзакций
plt.hist(
    df[df["Class"] == 0]["Amount"],
    bins=50,
    alpha=0.5,
    label="Обычные"
)

plt.hist(
    df[df["Class"] == 1]["Amount"],
    bins=50,
    alpha=0.5,
    label="Мошеннические"
)

plt.title("Распределение сумм по типам транзакций")
plt.xlabel("Сумма транзакции")
plt.ylabel("Количество транзакций")
plt.legend()

plt.show()

#Распределение транзакций по времени
plt.hist(df["Time"], bins=50)

plt.title("Распределение транзакций по времени")
plt.xlabel("Время")
plt.ylabel("Количество транзакций")

plt.show()

#Распределение признака V1
plt.hist(df["V1"], bins=50)

plt.title("Распределение признака V1")
plt.xlabel("Значение V1")
plt.ylabel("Количество транзакций")

plt.show()