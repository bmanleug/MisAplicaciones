# Archivo base para el despliegue del Agente en Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Retención y Uplift - Starbucks Rewards",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Aplicación Interactiva: T-learner, Curva Qini y Políticas de Retención")
st.markdown("""
Esta aplicación permite visualizar el modelo de retención, las políticas de decisión del **Paso 3** 
y el rendimiento del modelo a través de una interfaz interactiva en Streamlit.
""")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/uplift_experiment.csv")
    except FileNotFoundError:
        # Datos sintéticos de respaldo si no se encuentra el archivo localmente
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "customer_id": range(1, n + 1),
            "treatment_group": np.random.choice(["treatment", "control"], size=n, p=[0.5, 0.5]),
            "recency_days": np.random.randint(0, 200, size=n),
            "frequency": np.random.randint(1, 30, size=n),
            "monetary": np.random.uniform(10, 500, size=n),
            "email_open_rate": np.random.uniform(0, 1, size=n),
            "responded_90d": np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
            "value_90d": np.random.choice([0, 50], size=n, p=[0.6, 0.4])
        })
    return df

df = load_data()

# --- BARRA LATERAL: CONTROLES INTERACTIVOS ---
st.sidebar.header("🎛️ Controles Interactivos")
pct_contacto = st.sidebar.slider(
    "Porcentaje objetivo a contactar (%):",
    min_value=5, max_value=60, value=20, step=5,
    help="Define el porcentaje superior de clientes con mayor propensión/riesgo a contactar."
)

# --- ENTRENAMIENTO DEL T-LEARNER (SIMPLIFICADO) ---
features = ["recency_days", "frequency", "monetary", "email_open_rate"]
tratados = df[df["treatment_group"] == "treatment"]
control = df[df["treatment_group"] == "control"]

model_t = LogisticRegression(max_iter=1000).fit(tratados[features], tratados["responded_90d"])
model_c = LogisticRegression(max_iter=1000).fit(control[features], control["responded_90d"])

# Calcular probabilidades estimadas para todo el dataset
df["p_treatment"] = model_t.predict_proba(df[features])[:, 1]
df["p_control"] = model_c.predict_proba(df[features])[:, 1]
df["uplift"] = df["p_treatment"] - df["p_control"]

# --- SECCIÓN 1: TABLA DE POLÍTICAS DEL PASO 3 ---
st.header("📋 Paso 3 — Tabla de Políticas y Segmentación")
st.markdown("Evaluación del rendimiento y precisión según el porcentaje de clientes seleccionados para campaña.")

y_arr = np.asarray(df["responded_90d"])
tasa_base = y_arr.mean()
orden = np.argsort(df["uplift"].values)[::-1] # Ordenar de mayor a menor uplift

filas = []
for pct in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
    k = int(len(y_arr) * pct)
    y_top = y_arr[orden[:k]]
    precision_k = y_top.mean() if len(y_top) > 0 else 0
    recall_k = y_top.sum() / y_arr.sum() if y_arr.sum() > 0 else 0
    filas.append({
        "% Contactado": f"{pct:.0%}",
        "Clientes (k)": k,
        "Precision@k": round(precision_k, 3),
        "Recall@k": round(recall_k, 3),
        "Lift@k": round(precision_k / tasa_base, 2) if tasa_base > 0 else 0,
    })

policy_table = pd.DataFrame(filas)
st.dataframe(policy_table, use_container_width=True)

# --- SECCIÓN 2: CURVA QINI / GRÁFICA DE UPLIFT ---
st.header("📈 Curva Qini / Impacto del T-learner")
st.markdown(f"Visualización acumulada del beneficio de aplicar la política interactiva seleccionada (**{pct_contacto}%** de contacto).")

fig, ax = plt.subplots(figsize=(10, 5))

# Cálculo de la curva Qini acumulada simulada
sorted_df = df.iloc[orden].copy()
sorted_df["cumulative_gain"] = (sorted_df["responded_90d"] * (sorted_df["treatment_group"] == "treatment")).cumsum()
ax.plot(range(len(sorted_df)), sorted_df["cumulative_gain"], color="#00704A", linewidth=2.5, label="Modelo T-learner (Uplift)")
ax.plot([0, len(sorted_df)], [0, sorted_df["responded_90d"].sum()], color="#9A988E", linestyle="--", label="Aleatorio")

# Línea vertical interactiva según el control deslizante
k_interactivo = int(len(df) * (pct_contacto / 100.0))
ax.axvline(x=k_interactivo, color="#C4512F", linestyle=":", linewidth=2, label=f"Corte Interactivo ({pct_contacto}%)")

ax.set_title("Curva Qini Acumulada", fontsize=14, fontweight="bold")
ax.set_xlabel("Número de Clientes Contactados (Ordenados por Uplift)")
ax.set_ylabel("Ganancia Acumulada de Conversiones")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# --- MÉTRICAS RESUMIDAS ADICIONALES ---
col1, col2, col3 = st.columns(3)
col1.metric("Clientes Totales analizados", f"{len(df):,}")
col2.metric("Uplift Promedio Estimado", f"{df['uplift'].mean():.4f}")
col3.metric("Clientes en el umbral seleccionado", f"{k_interactivo:,}")