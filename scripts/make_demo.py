from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(42)

N_CUSTOMERS, N_NORMAL, N_ANOMALY = 200, 4900, 100

CATEGORIES = ["Продукты", "Транспорт", "Кафе", "Одежда",
              "Аптека", "Развлечения", "Электроника", "Путешествия"]
CAT_WEIGHTS = [0.30, 0.15, 0.20, 0.10, 0.10, 0.10, 0.03, 0.02]
MERCHANTS = {c: [f"{c}, точка {i}" for i in range(1, 4)] for c in CATEGORIES}
FOREIGN = ["DE", "TR", "AE", "US", "CN"]
UNKNOWN_MERCHANTS = [f"Неизвестный продавец {i}" for i in range(1, 11)]

customer_logmean = rng.normal(4.5, 0.5, N_CUSTOMERS + 1)  # у каждого клиента свой обычный чек


def make_times(days, hours):
    base = pd.Timestamp("2026-10-01")
    return (base + pd.to_timedelta(days, unit="D") + pd.to_timedelta(hours, unit="h")
            + pd.to_timedelta(rng.integers(0, 3600, len(days)), unit="s"))


# --- обычные операции ---
cust = rng.integers(1, N_CUSTOMERS + 1, N_NORMAL)
cat = rng.choice(CATEGORIES, N_NORMAL, p=CAT_WEIGHTS)
normal = pd.DataFrame({
    "Time": make_times(rng.integers(0, 30, N_NORMAL),
                       np.clip(rng.normal(14, 4, N_NORMAL), 0, 23).astype(int)),
    "Amount": np.exp(rng.normal(customer_logmean[cust], 0.6)).round(2),
    "Country": rng.choice(["RU", "KZ", "BY"], N_NORMAL, p=[0.93, 0.04, 0.03]),
    "Merchant": [rng.choice(MERCHANTS[c]) for c in cat],
    "Category": cat,
    "CustomerID": cust,
    "is_fraud": 0,
})

# --- заложенные аномалии трёх типов ---
kind = rng.integers(0, 3, N_ANOMALY)
cust_a = rng.integers(1, N_CUSTOMERS + 1, N_ANOMALY)
anom = pd.DataFrame({
    # 0: очень крупная сумма; 1: ночью и в чужой стране; 2: крупная сумма ночью
    "Time": make_times(rng.integers(0, 30, N_ANOMALY),
                       np.where(kind == 0, rng.integers(9, 21, N_ANOMALY),
                                rng.integers(0, 5, N_ANOMALY))),
    "Amount": np.where(kind == 1, rng.uniform(300, 2000, N_ANOMALY),
                       rng.uniform(3000, 15000, N_ANOMALY)).round(2),
    "Country": np.where(kind == 0, "RU", rng.choice(FOREIGN, N_ANOMALY)),
    "Merchant": np.where(kind == 0, rng.choice(MERCHANTS["Электроника"], N_ANOMALY),
                         rng.choice(UNKNOWN_MERCHANTS, N_ANOMALY)),
    "Category": np.where(kind == 0, "Электроника", "Путешествия"),
    "CustomerID": cust_a,
    "is_fraud": 1,
})

df = pd.concat([normal, anom], ignore_index=True).sort_values("Time").reset_index(drop=True)
df.insert(0, "ID", np.arange(1, len(df) + 1))
df["Time"] = df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

out = ROOT / "data" / "demo"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "transactions_demo.csv", index=False)
print(f"Сохранено: {out / 'transactions_demo.csv'}  ({len(df)} строк, аномалий: {df['is_fraud'].sum()})")