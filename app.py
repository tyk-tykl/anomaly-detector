import time

import altair as alt
import pandas as pd
import streamlit as st

from src.evaluate import calc_metrics, compare_methods, detect_anomalies, get_labels, save_comparison
from src.models import METHODS

SHOW_ROWS = 1000  # сколько аномалий показывать в таблице
MAX_POINTS = 2000   # сколько точек каждого типа рисовать на графике сумм
COLORS = alt.Scale(domain=["Обычные", "Аномалии"], range=["#4388D6", "#E05252"])


def show_single_result(res):
    result, metrics = res["result"], res["metrics"]
    anomalies = result[result["Anomaly"] == 1]

    st.success(f"Анализ завершён: {res['method']} (расчёт занял {res['seconds']:.1f} сек.)")

    if metrics:
        st.subheader("Оценка качества")
        st.caption("Метрики на всём файле (модель видела эти данные при обучении). "
                   "Честную оценку смотрите в «Сравнить все алгоритмы».")
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{metrics['Precision']:.4f}")
        c2.metric("Recall", f"{metrics['Recall']:.4f}")
        c3.metric("F1-score", f"{metrics['F1-score']:.4f}")
    else:
        st.info("Для оценки качества нужен столбец Class со значениями 0 и 1.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Всего транзакций", len(result))
    c2.metric("Найдено аномалий", len(anomalies))
    c3.metric("Доля аномалий", f"{len(anomalies) / len(result) * 100:.2f}%")

    st.subheader("Найденные аномалии")
    if len(anomalies) > SHOW_ROWS:
        st.caption(f"Показаны первые {SHOW_ROWS} из {len(anomalies)} записей. "
                   "Полный список можно скачать в CSV.")
    st.dataframe(anomalies.head(SHOW_ROWS), width="stretch")

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
        file_name="anomalies.csv",
        mime="text/csv",
        on_click="ignore",  # не перезапускать страницу при скачивании
    )


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
            result = detect_anomalies(df, method)
            seconds = time.perf_counter() - start

            csv = (
                result[result["Anomaly"] == 1]
                .to_csv(index=False, sep=";", decimal=",", float_format="%.8g")
                .encode("utf-8-sig")
            )
        y = get_labels(df)
        metrics = calc_metrics(y, result["Anomaly"]) if y is not None else None
        st.session_state["single"] = {
            "method": method, "result": result, "metrics": metrics,
            "seconds": seconds, "csv": csv,
        }
    except Exception as e:
        st.error(f"Ошибка при анализе: {e}")

if b2.button("Сравнить все алгоритмы"):
    try:
        with st.spinner("Сравниваю алгоритмы..."):
            comp = compare_methods(df)
            save_comparison(comp[0])
            st.session_state["comparison"] = comp
    except Exception as e:
        st.error(f"Ошибка при сравнении: {e}")

if "single" in st.session_state:
    show_single_result(st.session_state["single"])

if "comparison" in st.session_state:
    show_comparison(st.session_state["comparison"])