import io

import altair as alt
import pandas as pd
import streamlit as st

from src.evaluate import calc_metrics, compare_methods, detect_anomalies, get_labels
from src.models import METHODS

COLORS = alt.Scale(domain=["Обычные", "Аномалии"], range=["#4388D6", "#E05252"])


@st.cache_data
def load_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data))


def show_single_result(res):
    result, metrics = res["result"], res["metrics"]
    anomalies = result[result["Anomaly"] == 1]

    st.success(f"Анализ завершён: {res['method']}")

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
    st.dataframe(anomalies, width="stretch")

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
        if len(data) > 10000:  # прореживаем обычные точки, чтобы график не тормозил
            normal = data[data["Anomaly"] == 0].sample(5000, random_state=42)
            data = pd.concat([normal, data[data["Anomaly"] == 1]])
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
        "Скачать результаты CSV",
        result.to_csv(index=False).encode("utf-8-sig"),
        file_name="anomaly_results.csv",
        mime="text/csv",
    )


def show_comparison(comp):
    st.subheader("Сравнение алгоритмов")
    st.dataframe(comp, width="stretch")

    st.subheader("Время работы")
    st.bar_chart(comp.set_index("Алгоритм")["Время (сек.)"])

    metric_cols = ["Precision", "Recall", "F1-score"]
    if all(c in comp.columns for c in metric_cols):
        melted = comp.melt(id_vars="Алгоритм", value_vars=metric_cols,
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

try:
    df = load_csv(uploaded.getvalue())
except Exception as e:
    st.error(f"Не удалось прочитать файл: {e}")
    st.stop()

# сбрасываем старые результаты при загрузке нового файла
if st.session_state.get("file_id") != uploaded.file_id:
    st.session_state.clear()
    st.session_state["file_id"] = uploaded.file_id

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
            result = detect_anomalies(df, method)
        y = get_labels(df)
        metrics = calc_metrics(y, result["Anomaly"]) if y is not None else None
        st.session_state["single"] = {"method": method, "result": result, "metrics": metrics}
    except Exception as e:
        st.error(f"Ошибка при анализе: {e}")

if b2.button("Сравнить все алгоритмы"):
    try:
        with st.spinner("Сравниваю алгоритмы..."):
            st.session_state["comparison"] = compare_methods(df)
    except Exception as e:
        st.error(f"Ошибка при сравнении: {e}")

# результаты берутся из session_state, поэтому не пропадают при любом действии на странице
if "single" in st.session_state:
    show_single_result(st.session_state["single"])

if "comparison" in st.session_state:
    show_comparison(st.session_state["comparison"])