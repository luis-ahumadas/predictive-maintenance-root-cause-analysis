import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de página
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("🏭 Sistema Predictivo y Diagnóstico de Fallas")

# Cargar modelo y datos
@st.cache_resource
def load_artifacts():
    model = joblib.load('../models/best_model.pkl')
    X_original = pd.read_csv('../data/ai4i2020.csv')
    return model, X_original

model, df_original = load_artifacts()

# Sidebar para entrada de datos
st.sidebar.header("📊 Entrada de Datos de Sensor")

def user_input_features():
    # Valores por defecto - operación normal
    air_temp = st.sidebar.slider('Temperatura del Aire (K)', 295, 310, 300)
    process_temp = st.sidebar.slider('Temperatura del Proceso (K)', 305, 320, 310)
    rotational_speed = st.sidebar.slider('Velocidad de Rotación (RPM)', 1000, 3000, 1500)
    torque = st.sidebar.slider('Torque (Nm)', 1.0, 80.0, 30.0)
    tool_wear = st.sidebar.slider('Desgaste de Herramienta (min)', 0, 300, 100)
    product_type = st.sidebar.selectbox('Tipo de Producto', ['L', 'M', 'H'])

    data = {
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Type': product_type
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Procesar entrada (replicamos feature engineering del notebook)
@st.cache_data
def process_features(df):
    df_processed = df.copy()
    df_processed['Temp_Diff'] = df_processed['Process temperature [K]'] - df_processed['Air temperature [K]']
    df_processed['Rotational_speed_rads'] = df_processed['Rotational speed [rpm]'] * (2 * np.pi / 60)
    df_processed['Power'] = df_processed['Torque [Nm]'] * df_processed['Rotational_speed_rads']
    df_processed['Thermal_Efficiency'] = (df_processed['Air Temperature [K]'] / df_processed['Process temperature [K]']) * 100
    df_processed['Wear_Torque_Interaction'] = df_processed['Tool wear [min]'] * df_processed['Torque [Nm]']

    # Estadísticas por tipo
    type_avg_speed = df_original.groupby('Type')['Rotational speed [rpm]'].mean().to_dict()
    df_processed['Speed_Deviation'] = (df_processed['Rotational speed [rpm]'] -
                                       df_processed['Type'].map(type_avg_speed)) / df_processed['Type'].map(type_avg_speed)

    # One-hot encoding
    df_processed = pd.get_dummies(df_processed, columns=['Type'], prefix='Type')

    # Asegurar columnas
    expected_cols = ['Air temperature [K]', 'Process temperature [K]',
                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                     'Temp_Diff', 'Power', 'Thermal_Efficiency',
                     'Wear_Torque_Interaction', 'Speed_Deviation', 'Type_H',
                     'Type_L', 'Type_M']

    for col in expected_cols:
        if col not in df_processed.columns:
            df_processed[col] = 0

    return df_processed[expected_cols]

processed_input = process_features(input_df)

# Predicción
with st.spinner('Analizando datos...'):
    prediction = model.predict(processed_input)
    probability = model.predict_proba(processed_input)[0, 1]

    # SHAP para explicación
    classifier = model.named_steps['classifier']
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(processed_input)

# Mostrar resultados
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🔮 Predicción", "FALLO" if prediction[0] == 1 else "OPERACIÓN NORMAL")

with col2:
    st.metric("📈 Probabilidad de Fallo", f"{probability:.1%}")

with col3:
    status_color = "🔴" if prediction[0] == 1 else "🟢"
    st.metric("Estado", f"{status_color} {'Alerta' if prediction[0] == 1 else 'OK'}")

# Gráfico SHAP
if prediction[0] == 1:
    st.subheader("🔍 Análisis de Causa Raíz")

    # Visualización SHAP
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                         base_values=explainer.expected_value,
                                         data=processed_input.values[0],
                                         feature_names=processed_input.columns),
                                         show=False)
    st.pyplot(fig)

    # Diagnóstico automático
    st.subheader("📋 Diagnóstico Automático")

    # Reglas simples - versión simplificada del diagnóstico
    top_shap_idx = np.argmax(np.abs(shap_values[0]))
    top_feature = processed_input.columns[top_shap_idx]
    top_value = processed_input.values[0][top_shap_idx]

    if top_feature == 'Power' and shap_values[0][top_shap_idx] > 0:
        st.warning("**Causa Probable: PWF - Fallo de Potencia**")
        st.write(f"La potencia es alta ({top_value:.2f}), lo que indica posible sobrecarga del motor.")
    elif top_feature == 'Tool wear [min]' and shap_values[0][top_shap_idx] > 0:
        st.warning("**Causa Probable: TWF - Fallo por Desgaste de Herramienta**")
        st.write(f"El desgaste de la herramienta es crítico ({top_value:.0f} min), sugiere reemplazo inmediato.")
    elif top_feature == 'Wear_Torque_Interaction' and shap_values[0][top_shap_idx] > 0:
        st.warning("**Causa Probable: OSF - Fallo por Sobreesfuerzo**")
        st.write(f"Alta interacción entre desgaste y torque ({top_value:.2f}), la herramienta está sobreesforzada.")
    elif 'Temp' in top_feature and shap_values[0][top_shap_idx] < 0:
        st.warning("**Causa Probable: HDF - Fallo por Disipación de Calor**")
        st.write("Baja diferencia térmica, el sistema no está disipando calor eficientemente.")
    else:
        st.info("**Causa Probable: RNF - Fallo Aleatorio**")
        st.write("El patrón de fallo no sigue una causa específica identificable.")
    
    st.write(f"**Confianza del diagnóstico:** {np.abs(shap_values[0][top_shap_idx]) / np.sum(np.abs(shap_values[0])):.1%}")

else:
    st.success("✅ El equipo está operando dentro de parámetros normales.")

    st.subheader("📊 Monitoreo de Variables Clave")
    fig, ax = plt.subplots(figsize=(10, 4))
    values = processed_input.values[0]
    shap_summary = pd.DataFrame({
        'Feature': processed_input.columns,
        'Value': values
    })
    sns.barplot(data=shap_summary.sort_values('Value', ascending=False).head(8),
                x='Value', y='Feature', palette='viridis')
    ax.set_title('Valores Actuales vs. Rangos Operativos')
    st.pyplot(fig)

# Historial de predicciones (simulado)
st.subheader("📈 Tendencia de Probabilidad de Fallo")
historical_probs = np.random.beta(2, 10, 30) + probability * 0.3
historical_probs = np.clip(historical_probs, 0, 1)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(len(historical_probs)), historical_probs, marker='o',
        linewidth=2, color='red')
ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Umbral de alerta')
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Umbral de precaución')
ax.set_xlabel('Muestra (Horas)')
ax.set_ylabel('Probabilidad de Fallo')
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.caption("Dashboard desarrollado para el proyecto de Mantenimiento Predictivo con Análisis de Causa")
st.caption("Creado por Luis Ahumada")