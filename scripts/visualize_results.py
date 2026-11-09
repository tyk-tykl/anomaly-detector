from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
csv_path = RESULTS_DIR / "comparison.csv"

if not csv_path.exists():
    raise SystemExit(
        "Нет файла results/comparison.csv. Сначала запустите приложение "
        "и нажмите «Сравнить все алгоритмы»."
    )

df = pd.read_csv(csv_path)
needed = ["Алгоритм", "Precision", "Recall", "F1-score"]
if not all(c in df.columns for c in needed):
    raise SystemExit("В comparison.csv нет метрик. Нужен файл с столбцом Class.")

# График 1. F1-score
plt.figure(figsize=(10, 6))
plt.bar(df["Алгоритм"], df["F1-score"])
plt.title("Сравнение алгоритмов по F1-score")
plt.xlabel("Алгоритм")
plt.ylabel("F1-score")
plt.ylim(0, df["F1-score"].max() * 1.2)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "results_comparison.png", dpi=300)
plt.show()

# График 2. Precision и Recall
x = range(len(df))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar([i - width / 2 for i in x], df["Precision"], width=width, label="Precision")
plt.bar([i + width / 2 for i in x], df["Recall"], width=width, label="Recall")
plt.title("Сравнение Precision и Recall")
plt.xlabel("Алгоритм")
plt.ylabel("Значение метрики")
plt.xticks(list(x), df["Алгоритм"], rotation=15)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "precision_recall_comparison.png", dpi=300)
plt.show()

# График 3. Время работы
plt.figure(figsize=(10, 6))
plt.bar(df["Алгоритм"], df["Время (сек.)"])
plt.title("Время работы алгоритмов")
plt.xlabel("Алгоритм")
plt.ylabel("Секунды")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "time_comparison.png", dpi=300)
plt.show()