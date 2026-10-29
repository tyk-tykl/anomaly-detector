import pandas as pd
import matplotlib.pyplot as plt

# Результаты сравнения алгоритмов
results = {
    "Метод": [
        "Z-score",
        "Isolation Forest",
        "LOF",
        "One-Class SVM"
    ],
    "Precision": [
        0.012658,
        0.039106,
        0.021341,
        0.029940
    ],
    "Recall": [
        0.125000,
        0.875000,
        0.875000,
        0.937500
    ],
    "F1": [
        0.022989,
        0.074866,
        0.041667,
        0.058027
    ]
}

df = pd.DataFrame(results)

# График 1. Сравнение F1
plt.figure(figsize=(10, 6))

plt.bar(df["Метод"], df["F1"])

plt.title("Сравнение алгоритмов по F1-score")
plt.xlabel("Алгоритм")
plt.ylabel("F1-score")
plt.ylim(0, 0.09)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("../results/results_comparison.png", dpi=300)
plt.show()

# График 2. Сравнение Precision и Recall
x = range(len(df))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    df["Precision"],
    width=width,
    label="Precision"
)

plt.bar(
    [i + width / 2 for i in x],
    df["Recall"],
    width=width,
    label="Recall"
)

plt.title("Сравнение Precision и Recall")
plt.xlabel("Алгоритм")
plt.ylabel("Значение метрики")
plt.xticks(list(x), df["Метод"], rotation=15)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()

plt.savefig("../results/precision_recall_comparison.png", dpi=300)
plt.show()