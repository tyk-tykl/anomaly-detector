import time

import altair as alt
import pandas as pd
import streamlit as st
import io

from src.evaluate import (calc_metrics, compare_methods, confusion_table,
                          detect_anomalies, get_labels, save_comparison)
from src.models import METHODS

SHOW_ROWS = 1000  # сколько аномалий показывать в таблице
MAX_POINTS = 2000   # сколько точек каждого типа рисовать на графике сумм
FULL_CSV_MAX = 100000  # полный файл предлагаем скачать, если строк не больше
COLORS = alt.Scale(domain=["Обычные", "Аномалии"], range=["#4388D6", "#E05252"])
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def to_xlsx(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def show_single_result(res):
    result, metrics = res["result"], res["metrics"]
    anomalies = result[result["Anomaly"] == 1].sort_values("Score", ascending=False)

    st.success(f"Анализ завершён: {res['method']} (расчёт занял {res['seconds']:.1f} сек.)")

    if metrics:
        st.subheader("Оценка качества")
        st.caption("Метрики на всём файле (модель видела эти данные при обучении). "
                   "Честную оценку смотрите в «Сравнить все алгоритмы».")
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{metrics['Precision']:.4f}")
        if res.get("confusion") is not None:
            st.write("Матрица ошибок")
            st.dataframe(res["confusion"], width="stretch")
        c2.metric("Recall", f"{metrics['Recall']:.4f}")
        c3.metric("F1-score", f"{metrics['F1-score']:.4f}")
    else:
        st.info("Для оценки качества выберите в боковой панели столбец с меткой (значения 0 и 1).")

    c1, c2, c3 = st.columns(3)
    c1.metric("Всего транзакций", len(result))
    c2.metric("Найдено аномалий", len(anomalies))
    c3.metric("Доля аномалий", f"{len(anomalies) / len(result) * 100:.2f}%")

    st.subheader("Найденные аномалии")
    st.caption("Отсортированы по Score: чем выше, тем необычнее запись. Шкала Score у разных "
               "алгоритмов своя, сравнивать её между ними нельзя. В «Почему аномально» указаны "
               "три признака, сильнее всего отклонившихся от среднего (в стандартных отклонениях σ). "
               "Это приближённое объяснение, а не внутренняя логика модели.")
    if len(anomalies) > SHOW_ROWS:
        st.caption(f"Показаны первые {SHOW_ROWS} из {len(anomalies)} записей. "
                   "Полный список можно скачать в CSV.")
    first = ["Score", "Почему аномально"]
    cols = first + [c for c in anomalies.columns if c not in first]
    st.dataframe(anomalies[cols].head(SHOW_ROWS), width="stretch")

    st.subheader("Визуализация")
    counts = pd.DataFrame({
        "Тип": ["Обычные", "Аномалии"],
        "Количество": [(result["Anomaly"] == 0).sum(), (result["Anomaly"] == 1).sum()],
    })
    st.altair_chart(
        alt.Chart(counts).mark_bar().encode(
            x=alt.X("Тип:N"), y=alt.Y("Количество:Q"),
            color=alt.Color("Тип:N", scale=COLORS, legend=None),
            tooltip=["Тип", "Количество"],
        ).properties(title="Обычные транзакции и аномалии", height=300),
        width="stretch",
    )

    if "Amount" in result.columns:
        data = result[["Amount", "Anomaly"]].copy()
        data["Тип"] = data["Anomaly"].map({0: "Обычные", 1: "Аномалии"})
        data = pd.concat([
            g.sample(min(len(g), MAX_POINTS), random_state=42)
            for _, g in data.groupby("Anomaly")
        ])
        st.altair_chart(
            alt.Chart(data).mark_circle(opacity=0.6, size=45).encode(
                x=alt.X("Amount:Q", title="Сумма транзакции", scale=alt.Scale(zero=False)),
                y=alt.Y("jitter:Q", axis=None),
                color=alt.Color("Тип:N", scale=COLORS),
                tooltip=["Amount:Q", "Тип:N"],
            ).transform_calculate(jitter="random()")
            .properties(title="Распределение сумм транзакций", height=300),
            width="stretch",
        )

    st.download_button(
        "Скачать найденные аномалии (CSV)",
        res["csv"],
        file_name="anomalies.xlsx",
        mime=XLSX_MIME,
        key="dl_anomalies",  # не перезапускать страницу при скачивании
    )

    if res.get("full_csv") is not None:
        st.caption(f"Размер подготовленного файла: {len(res['full_csv']) / 1024:.0f} КБ")

    if res.get("full_csv") is not None:
        st.download_button(
            "Скачать весь файл с Score и Anomaly (CSV)",
            res["full_csv"],
            file_name="all_transactions_scored.xlsx",
            mime=XLSX_MIME,
            key="dl_full",
        )
    else:
        st.caption(f"Файл больше {FULL_CSV_MAX} строк, полная выгрузка отключена: "
                   "скачайте список аномалий.")


def show_comparison(comp):
    table, curves, baseline = comp

    st.subheader("Сравнение алгоритмов")
    st.dataframe(table, width="stretch")

    if "F1-score ±" in table.columns:
        st.caption("Метрики усреднены по 5 запускам с разным разбиением данных, "
                   "столбцы со знаком ± показывают стандартное отклонение. "
                   "Precision@1% и Recall@1%: качество среди 1% транзакций "
                   "с самым высоким score.")

    st.download_button(
        "Скачать таблицу сравнения (CSV)",
        table.round(4).to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        file_name="comparison.csv",
        mime="text/csv",
        on_click="ignore",
    )

    st.subheader("Время работы")
    st.bar_chart(table.set_index("Алгоритм")["Время (сек.)"])

    metric_cols = ["Precision", "Recall", "F1-score"]
    if all(c in table.columns for c in metric_cols):
        melted = table.melt(id_vars="Алгоритм", value_vars=metric_cols,
                            var_name="Метрика", value_name="Значение")
        st.altair_chart(
            alt.Chart(melted).mark_bar().encode(
                x="Алгоритм:N", xOffset="Метрика:N",
                y=alt.Y("Значение:Q", scale=alt.Scale(domain=[0, 1])),
                color="Метрика:N",
                tooltip=["Алгоритм", "Метрика", "Значение"],
            ).properties(title="Качество на проверочной выборке", height=350),
            width="stretch",
        )

    if not curves.empty:
        st.subheader("PR-кривая")
        st.caption(
            "Чем выше и правее кривая, тем лучше алгоритм ранжирует транзакции: "
            "самые подозрительные у него действительно чаще оказываются мошенническими. "
            "Пунктир показывает долю мошенничества в данных, то есть качество "
            "случайного угадывания. Avg Precision в таблице - площадь под кривой."
        )
        lines = alt.Chart(curves).mark_line().encode(
            x=alt.X("Recall:Q", title="Recall (полнота)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Precision:Q", title="Precision (точность)", scale=alt.Scale(domain=[0, 1])),
            color="Алгоритм:N",
            order="Recall:Q",
            tooltip=["Алгоритм", "Recall:Q", "Precision:Q"],
        )
        base = alt.Chart(pd.DataFrame({"y": [baseline]})).mark_rule(
            strokeDash=[4, 4], color="gray"
        ).encode(y="y:Q")
        st.altair_chart((lines + base).properties(height=400), width="stretch")


# ---------- Страница ----------
st.set_page_config(page_title="Детектор аномалий", page_icon="🔎", layout="wide")
st.title("Детектор аномалий")
st.write("Обнаружение необычных транзакций статистическими методами и машинным обучением.")
st.divider()

method = st.sidebar.selectbox("Выберите алгоритм", METHODS)

use_log = st.sidebar.checkbox(
    "Логарифм суммы (log1p)", value=True,
    help="Суммы распределены асимметрично, логарифм делает их ближе к симметричным.")

uploaded = st.file_uploader("Выберите CSV-файл с транзакциями", type=["csv"])

if uploaded is None:
    st.session_state.clear()
    st.info("Загрузите CSV-файл, чтобы начать работу.")
    st.stop()

if st.session_state.get("file_id") != uploaded.file_id:
    st.session_state.clear()
    try:
        with st.spinner("Читаю файл..."):
            st.session_state["df"] = pd.read_csv(uploaded)
        st.session_state["file_id"] = uploaded.file_id
    except Exception as e:
        st.error(f"Не удалось прочитать файл: {e}")
        st.stop()

df = st.session_state["df"]

# ---------- Выбор столбцов ----------
st.sidebar.subheader("Столбцы")
all_cols = list(df.columns)
label_options = ["— нет —"] + all_cols
guess = next((c for c in all_cols
              if c.lower() in ("class", "is_fraud", "fraud", "label", "anomaly", "target")), None)
label_choice = st.sidebar.selectbox(
    "Столбец с меткой (необязательно)", label_options,
    index=label_options.index(guess) if guess else 0,
    help="Столбец, где 1 — аномалия, 0 — норма. Нужен только для подсчёта метрик, "
         "в признаки он не попадает.")
label_col = None if label_choice == "— нет —" else label_choice

candidates = [c for c in all_cols if c != label_col]
default_feats = [c for c in candidates if not c.lower().endswith("id")]
feature_cols = st.sidebar.multiselect(
    "Признаки для анализа", candidates, default=default_feats,
    key=f"features_{label_col}",
    help="Числа берутся как есть, даты превращаются в час и день недели, "
         "текстовые столбцы — в частоту встречаемости. Столбцы-идентификаторы лучше убрать.")
if not feature_cols:
    st.warning("Выберите хотя бы один признак в боковой панели.")
    st.stop()

st.success("Файл успешно загружен!")
c1, c2 = st.columns(2)
c1.metric("Строк", len(df))
c2.metric("Столбцов", len(df.columns))
st.dataframe(df.head(10), width="stretch")
st.divider()

st.subheader("Анализ транзакций")
b1, b2 = st.columns(2)

if b1.button("Запустить анализ"):
    try:
        with st.spinner("Идёт анализ..."):
            start = time.perf_counter()
            result = detect_anomalies(df, method, use_log, feature_cols, label_col)
            seconds = time.perf_counter() - start

            sorted_all = result.sort_values("Score", ascending=False)
            csv = to_xlsx(sorted_all[sorted_all["Anomaly"] == 1])
            full_csv = to_xlsx(sorted_all) if len(result) <= FULL_CSV_MAX else None

        y = get_labels(df, label_col)
        metrics = calc_metrics(y, result["Anomaly"]) if y is not None else None
        st.session_state["single"] = {
            "method": method, "result": result, "metrics": metrics,
            "confusion": confusion_table(y, result["Anomaly"]) if y is not None else None,
            "seconds": seconds, "csv": csv, "full_csv": full_csv,
        }
    except Exception as e:
        st.error(f"Ошибка при анализе: {e}")

if b2.button("Сравнить все алгоритмы"):
    try:
        with st.spinner("Сравниваю алгоритмы..."):
            comp = compare_methods(df, use_log=use_log,
                                   feature_cols=feature_cols, label_col=label_col)
            save_comparison(comp[0])
            st.session_state["comparison"] = comp
    except Exception as e:
        st.error(f"Ошибка при сравнении: {e}")

if "single" in st.session_state:
    show_single_result(st.session_state["single"])

if "comparison" in st.session_state:
    show_comparison(st.session_state["comparison"])