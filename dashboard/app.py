# 📁 dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Sistema de Mantenimiento Predictivo",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ESTILO PERSONALIZADO
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3D59;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5A7A9A;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .status-ok {
        color: #28a745;
        font-weight: 700;
    }
    .status-fail {
        color: #dc3545;
        font-weight: 700;
    }
    .diagnosis-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES DE LIMPIEZA DE NOMBRES
# ============================================================================
def clean_feature_names(df):
    """Limpia los nombres de las características eliminando caracteres especiales"""
    df_clean = df.copy()
    df_clean.columns = [
        col.replace('[', '').replace(']', '').replace('<', '').replace('>', '').strip()
        .replace(' ', '_')  # Reemplazar espacios por guiones bajos
        for col in df_clean.columns
    ]
    return df_clean

def standardize_feature_names(df):
    """Estandariza los nombres de características al formato usado por el modelo"""
    df_std = df.copy()
    
    # Mapeo de nombres comunes
    name_mapping = {
        'Air_temperature_K': 'Air_temperature_K',
        'Air temperature [K]': 'Air_temperature_K',
        'Process_temperature_K': 'Process_temperature_K',
        'Process temperature [K]': 'Process_temperature_K',
        'Rotational_speed_rpm': 'Rotational_speed_rpm',
        'Rotational speed [rpm]': 'Rotational_speed_rpm',
        'Torque_Nm': 'Torque_Nm',
        'Torque [Nm]': 'Torque_Nm',
        'Tool_wear_min': 'Tool_wear_min',
        'Tool wear [min]': 'Tool_wear_min',
        'Temp_Diff': 'Temp_Diff',
        'Power': 'Power',
        'Thermal_Efficiency': 'Thermal_Efficiency',
        'Wear_Torque_Interaction': 'Wear_Torque_Interaction',
        'Speed_Deviation': 'Speed_Deviation',
        'Type_H': 'Type_H',
        'Type_L': 'Type_L',
        'Type_M': 'Type_M'
    }
    
    # Renombrar columnas
    df_std.columns = [name_mapping.get(col, col) for col in df_std.columns]
    return df_std

# ============================================================================
# CARGA DE DATOS Y ENTRENAMIENTO DEL MODELO
# ============================================================================
@st.cache_resource
def load_or_train_model():
    """Carga el modelo guardado o entrena uno de demostración"""
    
    # Intentar cargar modelo guardado
    try:
        model = joblib.load('../models/best_model.pkl')
        feature_importance = pd.read_csv('../data/feature_importance.csv')
        df_original = pd.read_csv('../data/ai4i2020.csv')
        return model, feature_importance, df_original, True
    except (FileNotFoundError, OSError):
        st.warning("⚠️ Archivos de modelo no encontrados. Entrenando modelo de demostración...")
        
        try:
            # Cargar datos desde el directorio actual
            data_paths = [
                '../data/ai4i2020.csv',
                './data/ai4i2020.csv',
                './ai4i2020.csv'
            ]
            
            df_original = None
            for path in data_paths:
                try:
                    df_original = pd.read_csv(path)
                    break
                except FileNotFoundError:
                    continue
            
            if df_original is None:
                st.warning("📊 Datos originales no encontrados. Generando datos sintéticos...")
                df_original = generate_synthetic_data()
            
            # Entrenar modelo de demostración
            model, feature_importance = train_demo_model(df_original)
            st.success("✅ Modelo de demostración entrenado exitosamente.")
            return model, feature_importance, df_original, False
            
        except Exception as e:
            st.error(f"❌ Error al entrenar modelo: {str(e)}")
            # Crear modelo simple como último recurso
            model = create_fallback_model()
            feature_importance = pd.DataFrame({
                'feature': ['Torque_Nm', 'Tool_wear_min', 'Power', 'Rotational_speed_rpm', 'Temp_Diff'],
                'importance': [0.3, 0.25, 0.2, 0.15, 0.1]
            })
            df_original = generate_synthetic_data()
            return model, feature_importance, df_original, False

def generate_synthetic_data(n=1000):
    """Genera datos sintéticos para el modelo de demostración"""
    np.random.seed(42)
    
    # Variables base con distribuciones realistas
    air_temp = np.random.normal(300, 5, n)
    process_temp = air_temp + np.random.normal(10, 3, n)
    rotational_speed = np.random.uniform(1000, 3000, n)
    torque = np.random.uniform(5, 70, n)
    tool_wear = np.random.uniform(0, 300, n)
    product_type = np.random.choice(['L', 'M', 'H'], n)
    
    # Crear lógica de fallo
    power = torque * (rotational_speed * 2 * np.pi / 60)
    temp_diff = process_temp - air_temp
    
    # Probabilidad de fallo basada en condiciones
    prob_failure = (
        0.3 * (tool_wear / 300) +
        0.25 * (power / 5000) +
        0.2 * (1 / (temp_diff / 15)) +
        0.15 * (torque / 70) +
        0.1 * np.random.random(n)
    )
    prob_failure = np.clip(prob_failure, 0, 1)
    machine_failure = (prob_failure > 0.65).astype(int)
    
    # Modos de fallo
    twf = ((tool_wear > 200) & (np.random.random(n) < 0.7)).astype(int)
    hdf = ((temp_diff < 5) & (np.random.random(n) < 0.3)).astype(int)
    pwf = ((power > 4500) & (np.random.random(n) < 0.4)).astype(int)
    osf = ((torque > 55) & (np.random.random(n) < 0.3)).astype(int)
    rnf = (np.random.random(n) < 0.05).astype(int)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Type': product_type,
        'Machine failure': machine_failure,
        'TWF': twf,
        'HDF': hdf,
        'PWF': pwf,
        'OSF': osf,
        'RNF': rnf
    })
    
    return df

def train_demo_model(df):
    """Entrena un modelo de demostración con los datos disponibles"""
    
    # Feature engineering
    df_processed = df.copy()
    df_processed['Temp_Diff'] = df_processed['Process temperature [K]'] - df_processed['Air temperature [K]']
    df_processed['Rotational_speed_rads'] = df_processed['Rotational speed [rpm]'] * (2 * np.pi / 60)
    df_processed['Power'] = df_processed['Torque [Nm]'] * df_processed['Rotational_speed_rads']
    df_processed['Thermal_Efficiency'] = (df_processed['Air temperature [K]'] / df_processed['Process temperature [K]']) * 100
    df_processed['Wear_Torque_Interaction'] = df_processed['Tool wear [min]'] * df_processed['Torque [Nm]']
    
    # Estadísticas por tipo
    type_avg_speed = df_processed.groupby('Type')['Rotational speed [rpm]'].mean().to_dict()
    df_processed['Speed_Deviation'] = (df_processed['Rotational speed [rpm]'] - 
                                        df_processed['Type'].map(type_avg_speed)) / df_processed['Type'].map(type_avg_speed)
    
    # One-hot encoding
    df_processed = pd.get_dummies(df_processed, columns=['Type'], prefix='Type')
    
    # Seleccionar features
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                'Torque [Nm]', 'Tool wear [min]', 'Temp_Diff', 'Power', 
                'Thermal_Efficiency', 'Wear_Torque_Interaction', 'Speed_Deviation',
                'Type_H', 'Type_L', 'Type_M']
    
    # Asegurar columnas
    for col in features:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    X = df_processed[features]
    y = df_processed['Machine failure']
    
    # Limpiar nombres (usar el mismo formato que el modelo)
    X_clean = clean_feature_names(X)
    
    # Dividir
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entrenar modelo con SMOTE
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, sampling_strategy=0.3)),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Calcular importancia de características
    importance = pipeline.named_steps['classifier'].feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X_clean.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Guardar el modelo entrenado
    try:
        import os
        os.makedirs('../models', exist_ok=True)
        os.makedirs('../data', exist_ok=True)
        joblib.dump(pipeline, '../models/best_model.pkl')
        feature_importance.to_csv('../data/feature_importance.csv', index=False)
        df.to_csv('../data/ai4i2020.csv', index=False)
    except Exception as e:
        pass
    
    return pipeline, feature_importance

def create_fallback_model():
    """Crea un modelo simple para casos de emergencia"""
    return ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])

# ============================================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================================
def process_features(df, df_original=None):
    """Aplica feature engineering a los datos de entrada y estandariza nombres"""
    df_processed = df.copy()
    
    # Crear características derivadas
    df_processed['Temp_Diff'] = df_processed['Process temperature [K]'] - df_processed['Air temperature [K]']
    df_processed['Rotational_speed_rads'] = df_processed['Rotational speed [rpm]'] * (2 * np.pi / 60)
    df_processed['Power'] = df_processed['Torque [Nm]'] * df_processed['Rotational_speed_rads']
    df_processed['Thermal_Efficiency'] = (df_processed['Air temperature [K]'] / df_processed['Process temperature [K]']) * 100
    df_processed['Wear_Torque_Interaction'] = df_processed['Tool wear [min]'] * df_processed['Torque [Nm]']
    
    # Estadísticas por tipo de producto
    if df_original is not None:
        try:
            type_avg_speed = df_original.groupby('Type')['Rotational speed [rpm]'].mean().to_dict()
            df_processed['Speed_Deviation'] = (df_processed['Rotational speed [rpm]'] - 
                                                df_processed['Type'].map(type_avg_speed)) / df_processed['Type'].map(type_avg_speed)
        except:
            df_processed['Speed_Deviation'] = 0
    else:
        df_processed['Speed_Deviation'] = 0
    
    # One-hot encoding
    df_processed = pd.get_dummies(df_processed, columns=['Type'], prefix='Type')
    
    # Asegurar columnas
    expected_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
                     'Torque [Nm]', 'Tool wear [min]', 'Temp_Diff', 'Power', 
                     'Thermal_Efficiency', 'Wear_Torque_Interaction', 'Speed_Deviation',
                     'Type_H', 'Type_L', 'Type_M']
    
    for col in expected_cols:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    # Limpiar nombres para que coincidan con el modelo
    df_clean = clean_feature_names(df_processed[expected_cols])
    
    return df_clean

def diagnose_failure(shap_values, features, threshold=0.15):
    """Diagnostica la causa raíz basada en SHAP"""
    diagnosis = {
        'causa_principal': None,
        'confianza': 0.0,
        'evidencia': [],
        'recomendacion': None
    }
    
    if shap_values is None or len(shap_values) == 0:
        return diagnosis
    
    try:
        # Manejar diferentes formas de SHAP values
        if hasattr(shap_values, 'shape'):
            # Si es 3D: (n_samples, n_features, n_outputs)
            if len(shap_values.shape) == 3:
                # Tomar la primera muestra y la clase positiva (índice 1)
                shap_contrib = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
            # Si es 2D con 2 columnas: (n_features, 2) para clasificación binaria
            elif len(shap_values.shape) == 2 and shap_values.shape[1] == 2:
                shap_contrib = shap_values[:, 1]  # Tomar clase positiva
                # Si es una matriz de (1, 2), aplanar
                if len(shap_contrib.shape) > 1:
                    shap_contrib = shap_contrib.flatten()
            # Si es 2D normal: (n_samples, n_features)
            elif len(shap_values.shape) == 2:
                shap_contrib = shap_values[0] if shap_values.shape[0] > 0 else shap_values
            # Si es 1D: vector de features
            else:
                shap_contrib = shap_values
        else:
            shap_contrib = shap_values
        
        # Asegurar que shap_contrib es 1D
        if isinstance(shap_contrib, np.ndarray):
            if len(shap_contrib.shape) > 1:
                shap_contrib = shap_contrib.flatten()
        
        # Asegurar que features es un DataFrame y obtener sus valores
        if hasattr(features, 'values'):
            feature_values = features.values
            if len(feature_values.shape) > 1:
                feature_values = feature_values[0] if feature_values.shape[0] > 0 else feature_values
        else:
            feature_values = features
        
        # Asegurar que feature_values es 1D
        if isinstance(feature_values, np.ndarray) and len(feature_values.shape) > 1:
            feature_values = feature_values.flatten()
        
        # Obtener nombres de características
        if hasattr(features, 'columns'):
            feature_names = features.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(len(shap_contrib))]
        
        # Crear DataFrame asegurando dimensiones correctas
        if len(shap_contrib) == len(feature_names):
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'shap_value': shap_contrib,
                'value': feature_values[:len(feature_names)] if len(feature_values) >= len(feature_names) else feature_values
            })
        else:
            # Si hay discrepancia en dimensiones, ajustar
            min_len = min(len(shap_contrib), len(feature_names), len(feature_values))
            shap_df = pd.DataFrame({
                'feature': feature_names[:min_len],
                'shap_value': shap_contrib[:min_len],
                'value': feature_values[:min_len]
            })
        
        shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
        top_features = shap_df.sort_values('abs_shap', ascending=False).head(5)
        top_features = top_features[top_features['abs_shap'] > threshold]
        
        if len(top_features) == 0:
            diagnosis['causa_principal'] = 'RNF - Fallo Aleatorio'
            diagnosis['confianza'] = 0.3
            diagnosis['recomendacion'] = 'Realizar inspección general y monitorear variables.'
            return diagnosis
        
        # Reglas de diagnóstico
        for _, row in top_features.iterrows():
            feature = row['feature']
            value = row['value']
            shap_val = row['shap_value']
            
            if feature == 'Power' and shap_val > 0:
                diagnosis['causa_principal'] = 'PWF - Fallo de Potencia'
                diagnosis['evidencia'].append(f'Potencia alta ({value:.2f} W)')
                diagnosis['recomendacion'] = 'Revisar sistema eléctrico y condiciones de carga.'
                diagnosis['confianza'] = max(diagnosis['confianza'], 0.85)
                
            elif feature == 'Tool_wear_min' and shap_val > 0:
                diagnosis['causa_principal'] = 'TWF - Desgaste de Herramienta'
                diagnosis['evidencia'].append(f'Desgaste crítico ({value:.0f} min)')
                diagnosis['recomendacion'] = 'Reemplazar herramienta inmediatamente.'
                diagnosis['confianza'] = max(diagnosis['confianza'], 0.80)
                
            elif feature == 'Wear_Torque_Interaction' and shap_val > 0:
                diagnosis['causa_principal'] = 'OSF - Sobreesfuerzo'
                diagnosis['evidencia'].append(f'Interacción alta ({value:.2f})')
                diagnosis['recomendacion'] = 'Reducir velocidad o torque, revisar parámetros de corte.'
                diagnosis['confianza'] = max(diagnosis['confianza'], 0.75)
                
            elif 'Temp' in feature and shap_val < 0:
                diagnosis['causa_principal'] = 'HDF - Fallo por Disipación de Calor'
                diagnosis['evidencia'].append(f'Problema térmico detectado ({value:.2f}K)')
                diagnosis['recomendacion'] = 'Verificar sistema de refrigeración y ventilación.'
                diagnosis['confianza'] = max(diagnosis['confianza'], 0.70)
                
            elif 'Rotational' in feature and shap_val < 0:
                diagnosis['causa_principal'] = 'OSF - Sobreesfuerzo'
                diagnosis['evidencia'].append(f'Velocidad anormal ({value:.0f} RPM)')
                diagnosis['recomendacion'] = 'Ajustar velocidad a parámetros operativos.'
                diagnosis['confianza'] = max(diagnosis['confianza'], 0.65)
        
        if diagnosis['causa_principal'] is None:
            diagnosis['causa_principal'] = 'RNF - Fallo Aleatorio'
            diagnosis['confianza'] = 0.3
            diagnosis['recomendacion'] = 'Realizar inspección general y monitorear variables.'
            
    except Exception as e:
        diagnosis['causa_principal'] = 'Error en diagnóstico'
        diagnosis['confianza'] = 0
        diagnosis['recomendacion'] = f'Error: {str(e)}'
    
    return diagnosis

def generate_historical_data(current_prob, n_points=30):
    """Genera datos históricos simulados para la tendencia"""
    base_trend = np.linspace(current_prob * 0.5, current_prob, n_points)
    noise = np.random.normal(0, 0.05, n_points)
    historical = np.clip(base_trend + noise, 0, 1)
    return historical

# ============================================================================
# CARGA DEL MODELO
# ============================================================================
model, feature_importance, df_original, is_pretrained = load_or_train_model()

# ============================================================================
# INTERFAZ DE USUARIO
# ============================================================================
st.markdown('<div class="main-header">🏭 Sistema de Mantenimiento Predictivo</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predicción de fallos y diagnóstico de causa raíz para maquinaria industrial</div>', unsafe_allow_html=True)

# Sidebar - Entrada de datos
st.sidebar.markdown("## 📊 Datos del Sensor")
st.sidebar.markdown("Ajusta los valores para simular diferentes condiciones de operación:")

# Crear columnas para mejor organización
col1, col2 = st.sidebar.columns(2)

with col1:
    air_temp = st.slider('🌡️ Temp. Aire (K)', 295.0, 310.0, 300.0, step=0.1)
    process_temp = st.slider('🌡️ Temp. Proceso (K)', 305.0, 320.0, 310.0, step=0.1)
    rotational_speed = st.slider('🔄 Velocidad (RPM)', 1000, 3000, 1500, step=100)
    torque = st.slider('⚙️ Torque (Nm)', 1.0, 80.0, 30.0, step=0.5)

with col2:
    tool_wear = st.slider('🔧 Desgaste (min)', 0, 300, 100, step=5)
    product_type = st.selectbox('📦 Tipo Producto', ['L', 'M', 'H'])
    alert_threshold = st.slider('🚨 Umbral Alerta', 0.3, 0.9, 0.7, step=0.05)

# Añadir botón para valores extremos (casos de prueba)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Casos de Prueba")

if st.sidebar.button("⚠️ Fallo Típico (Desgaste)"):
    st.session_state.test_values = {
        'air_temp': 300.0,
        'process_temp': 308.0,
        'rotational_speed': 1500,
        'torque': 50.0,
        'tool_wear': 250,
        'product_type': 'M'
    }
    st.rerun()

if st.sidebar.button("🔴 Fallo Crítico (Potencia)"):
    st.session_state.test_values = {
        'air_temp': 300.0,
        'process_temp': 310.0,
        'rotational_speed': 2800,
        'torque': 65.0,
        'tool_wear': 200,
        'product_type': 'H'
    }
    st.rerun()

if st.sidebar.button("🟢 Operación Normal"):
    st.session_state.test_values = {
        'air_temp': 300.0,
        'process_temp': 310.0,
        'rotational_speed': 1500,
        'torque': 30.0,
        'tool_wear': 50,
        'product_type': 'M'
    }
    st.rerun()

# Aplicar valores de prueba si existen
if 'test_values' in st.session_state:
    air_temp = st.session_state.test_values['air_temp']
    process_temp = st.session_state.test_values['process_temp']
    rotational_speed = st.session_state.test_values['rotational_speed']
    torque = st.session_state.test_values['torque']
    tool_wear = st.session_state.test_values['tool_wear']
    product_type = st.session_state.test_values['product_type']

# ============================================================================
# PROCESAMIENTO DE DATOS
# ============================================================================
# Crear DataFrame de entrada
input_df = pd.DataFrame({
    'Air temperature [K]': [air_temp],
    'Process temperature [K]': [process_temp],
    'Rotational speed [rpm]': [rotational_speed],
    'Torque [Nm]': [torque],
    'Tool wear [min]': [tool_wear],
    'Type': [product_type]
})

# Procesar características y limpiar nombres
processed_input = process_features(input_df, df_original)

# Mostrar los nombres de las características para depuración
# st.write("Nombres de características procesadas:", processed_input.columns.tolist())

# ============================================================================
# PREDICCIÓN
# ============================================================================
if model is not None:
    try:
        # Verificar que los nombres de las características coincidan
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
            if hasattr(classifier, 'feature_names_in_'):
                expected_features = classifier.feature_names_in_
            else:
                expected_features = processed_input.columns
        else:
            expected_features = processed_input.columns
        
        # Asegurar que los nombres coinciden
        if list(processed_input.columns) != list(expected_features):
            try:
                processed_input = processed_input[expected_features]
            except KeyError:
                st.warning("⚠️ Reajustando nombres de características...")
                if len(processed_input.columns) == len(expected_features):
                    processed_input.columns = expected_features
        
        # Realizar predicción
        prediction = model.predict(processed_input)
        probability = model.predict_proba(processed_input)[0, 1]
        
        # Calcular SHAP (si es posible)
        try:
            if hasattr(classifier, 'feature_importances_'):
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(processed_input)
                
                # Imprimir info de debug (opcional)
                # st.write(f"SHAP shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'unknown'}")
            else:
                shap_values = None
        except Exception as e:
            st.warning(f"No se pudo calcular SHAP: {e}")
            shap_values = None
        
        # Diagnóstico - pasar solo el primer sample si es necesario
        if shap_values is not None and hasattr(shap_values, 'shape'):
            if len(shap_values.shape) == 3:
                # (n_samples, n_features, n_outputs)
                diagnosis = diagnose_failure(shap_values, processed_input)
            elif len(shap_values.shape) == 2 and shap_values.shape[1] == 2:
                # (n_samples, 2) para clasificación binaria
                diagnosis = diagnose_failure(shap_values, processed_input)
            elif len(shap_values.shape) == 2:
                # (n_samples, n_features)
                diagnosis = diagnose_failure(shap_values, processed_input)
            else:
                diagnosis = diagnose_failure(shap_values, processed_input)
        else:
            diagnosis = {'causa_principal': 'No disponible', 'confianza': 0, 'evidencia': [], 'recomendacion': 'Cargar modelo para análisis completo.'}
        
    except Exception as e:
        st.error(f"❌ Error en la predicción: {str(e)}")
        st.error(f"Características esperadas: {expected_features}")
        st.error(f"Características recibidas: {processed_input.columns.tolist()}")
        prediction = [0]
        probability = 0.1
        shap_values = None
        diagnosis = {'causa_principal': 'Error', 'confianza': 0, 'evidencia': [], 'recomendacion': 'Revisar configuración del modelo.'}
else:
    prediction = [0]
    probability = 0.15
    shap_values = None
    diagnosis = {'causa_principal': 'No disponible', 'confianza': 0, 'evidencia': [], 'recomendacion': 'Cargar modelo para análisis completo.'}

# ============================================================================
# MÉTRICAS PRINCIPALES
# ============================================================================
st.markdown("## 📊 Monitoreo en Tiempo Real")

# Asegurar que probability es un escalar
if isinstance(probability, (np.ndarray, list)):
    probability = probability[0] if len(probability) > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)

with col1:
    status = "⚠️ FALLO" if prediction[0] == 1 else "✅ OPERACIÓN NORMAL"
    status_class = "status-fail" if prediction[0] == 1 else "status-ok"
    st.markdown(f"""
    <div class="metric-card">
        <h4>Estado</h4>
        <h2 class="{status_class}">{status}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    prob_color = "#dc3545" if probability > alert_threshold else ("#ffc107" if probability > 0.4 else "#28a745")
    st.markdown(f"""
    <div class="metric-card">
        <h4>Probabilidad de Fallo</h4>
        <h2 style="color: {prob_color}">{probability:.1%}</h2>
        <small>Umbral: {alert_threshold:.0%}</small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    confidence = diagnosis.get('confianza', 0) if isinstance(diagnosis, dict) else 0
    st.markdown(f"""
    <div class="metric-card">
        <h4>Confianza Diagnóstico</h4>
        <h2>{confidence:.1%}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    temp_diff = process_temp - air_temp
    power = torque * (rotational_speed * 2 * np.pi / 60)
    st.markdown(f"""
    <div class="metric-card">
        <h4>Parámetros Críticos</h4>
        <div>ΔT: {temp_diff:.1f}K</div>
        <div>Potencia: {power:.1f}W</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# ALERTA Y DIAGNÓSTICO
# ============================================================================
st.markdown("---")

if prediction[0] == 1:
    st.markdown("## 🔍 Análisis de Causa Raíz")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        causa = diagnosis.get('causa_principal', 'No determinado') if isinstance(diagnosis, dict) else 'No determinado'
        confianza = diagnosis.get('confianza', 0) if isinstance(diagnosis, dict) else 0
        evidencia = diagnosis.get('evidencia', []) if isinstance(diagnosis, dict) else []
        recomendacion = diagnosis.get('recomendacion', 'Sin recomendación') if isinstance(diagnosis, dict) else 'Sin recomendación'
        
        st.markdown(f"""
        <div class="diagnosis-box">
            <h3>⚠️ {causa}</h3>
            <p><strong>Confianza:</strong> {confianza:.1%}</p>
            <p><strong>Evidencia:</strong></p>
            <ul>
                {"".join([f"<li>{e}</li>" for e in evidencia]) if evidencia else "<li>No se encontró evidencia específica</li>"}
            </ul>
            <p><strong>Recomendación:</strong> {recomendacion}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        # Gauge de riesgo
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Nivel de Riesgo"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#FF6B6B"},
                'steps': [
                    {'range': [0, 30], 'color': "#4CAF50"},
                    {'range': [30, 70], 'color': "#FFC107"},
                    {'range': [70, 100], 'color': "#FF6B6B"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': alert_threshold * 100
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("## ✅ Estado Operativo")
    st.markdown(f"""
    <div class="success-box">
        <h3>✅ Operación Normal</h3>
        <p>El equipo está funcionando dentro de los parámetros establecidos.</p>
        <p><strong>Probabilidad de fallo:</strong> {probability:.1%}</p>
        <p><strong>Recomendación:</strong> Monitoreo continuo. Mantener parámetros actuales.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# VISUALIZACIONES AVANZADAS
# ============================================================================
st.markdown("---")
st.markdown("## 📈 Análisis Visual")

# Crear pestañas para diferentes visualizaciones
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Análisis SHAP", "📊 Distribución", "📉 Tendencia", "🎯 Importancia"])

with tab1:
    if shap_values is not None and prediction[0] == 1:
        st.markdown("### Contribución de Características (SHAP)")
        try:
            # Verificar la forma de los SHAP values
            if hasattr(shap_values, 'shape'):
                if len(shap_values.shape) == 3:  # (n_samples, n_features, n_outputs)
                    # Para modelos multi-output, tomar la primera salida
                    shap_values_single = shap_values[0, :, 0]  # Primera muestra, todas las features, primera salida
                    
                    # Crear el objeto Explanation correctamente
                    if hasattr(explainer, 'expected_value'):
                        if isinstance(explainer.expected_value, (list, np.ndarray)):
                            expected_value = explainer.expected_value[0]  # Primera salida
                        else:
                            expected_value = explainer.expected_value
                    else:
                        expected_value = 0
                    
                    # Crear Explanation con la forma correcta
                    explanation = shap.Explanation(
                        values=shap_values_single,
                        base_values=expected_value,
                        data=processed_input.values[0],
                        feature_names=processed_input.columns.tolist()
                    )
                    
                    # Graficar waterfall
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.waterfall_plot(explanation, show=False)
                    st.pyplot(fig)
                    
                elif len(shap_values.shape) == 2 and shap_values.shape[1] == 2:
                    # Si tiene 2 columnas (para clasificación binaria)
                    shap_values_single = shap_values[0, :]  # Todas las features de la primera muestra
                    if isinstance(shap_values_single, np.ndarray) and len(shap_values_single.shape) > 1:
                        shap_values_single = shap_values_single[:, 1]  # Tomar la clase positiva (fallo)
                    
                    # Crear Explanation
                    if hasattr(explainer, 'expected_value'):
                        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                    else:
                        expected_value = 0
                    
                    explanation = shap.Explanation(
                        values=shap_values_single,
                        base_values=expected_value,
                        data=processed_input.values[0],
                        feature_names=processed_input.columns.tolist()
                    )
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.waterfall_plot(explanation, show=False)
                    st.pyplot(fig)
                    
                elif len(shap_values.shape) == 2:
                    # Si es 2D, asumir que es (n_samples, n_features)
                    shap_values_single = shap_values[0]
                    explanation = shap.Explanation(
                        values=shap_values_single,
                        base_values=explainer.expected_value if not isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value[0],
                        data=processed_input.values[0],
                        feature_names=processed_input.columns.tolist()
                    )
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.waterfall_plot(explanation, show=False)
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ Forma de SHAP values no esperada")
                
        except Exception as e:
            st.warning(f"No se pudo generar el gráfico SHAP: {e}")
        
        # Resumen de contribuciones (versión mejorada)
        st.markdown("#### Top Características que más influyen")
        
        # Manejar diferentes formas de SHAP values
        try:
            if hasattr(shap_values, 'shape'):
                if len(shap_values.shape) == 3:
                    shap_contrib = shap_values[0, :, 1]  # Primera muestra, clase positiva
                elif len(shap_values.shape) == 2 and shap_values.shape[1] == 2:
                    shap_contrib = shap_values[:, 1]  # Todas las muestras, clase positiva
                    shap_contrib = shap_contrib[0]  # Primera muestra
                elif len(shap_values.shape) == 2:
                    shap_contrib = shap_values[0]
                else:
                    shap_contrib = shap_values
            else:
                shap_contrib = shap_values
            
            shap_df = pd.DataFrame({
                'Característica': processed_input.columns,
                'Contribución': shap_contrib
            })
            shap_df['|Contribución|'] = np.abs(shap_df['Contribución'])
            shap_df = shap_df.sort_values('|Contribución|', ascending=False).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#FF6B6B' if x < 0 else '#4CAF50' for x in shap_df['Contribución']]
            ax.barh(shap_df['Característica'], shap_df['Contribución'], color=colors)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.set_xlabel('Contribución SHAP (Clase: Fallo)')
            ax.set_title('Top 10 Características - Contribución Individual para Fallo')
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"No se pudo generar el gráfico de contribuciones: {e}")
    else:
        st.info("ℹ️ Las explicaciones SHAP están disponibles cuando se detecta un fallo.")

with tab2:
    st.markdown("### Distribución de Variables Críticas")
    
    if df_original is not None:
        # Seleccionar variable para visualizar
        var_to_plot = st.selectbox(
            "Selecciona variable:",
            ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 
             'Torque [Nm]', 'Tool wear [min]']
        )
        
        # Crear gráfico con distribución y valor actual
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df_original[var_to_plot], bins=30, alpha=0.7, color='#4CAF50', edgecolor='black')
        ax.axvline(input_df[var_to_plot].values[0], color='red', linestyle='--', 
                   linewidth=2, label='Valor Actual')
        ax.set_xlabel(var_to_plot)
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Distribución de {var_to_plot}')
        ax.legend()
        st.pyplot(fig)
        
        # Estadísticas
        mean_val = df_original[var_to_plot].mean()
        std_val = df_original[var_to_plot].std()
        current_val = input_df[var_to_plot].values[0]
        z_score = (current_val - mean_val) / std_val if std_val > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Media", f"{mean_val:.2f}")
        col2.metric("Desviación Estándar", f"{std_val:.2f}")
        col3.metric("Z-Score", f"{z_score:.2f}")
    else:
        st.warning("⚠️ Datos originales no disponibles para visualización.")

with tab3:
    st.markdown("### Tendencia de Probabilidad de Fallo")
    st.markdown("*Simulación basada en el estado actual*")
    
    # Generar datos históricos
    historical = generate_historical_data(probability)
    
    # Crear gráfico de tendencia
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(historical))),
        y=historical,
        mode='lines+markers',
        name='Probabilidad',
        line=dict(color='#FF6B6B', width=2),
        marker=dict(size=6)
    ))
    
    # Añadir umbrales
    fig.add_hline(y=alert_threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Umbral Alerta ({alert_threshold:.0%})")
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                  annotation_text="Umbral Precaución (50%)")
    
    # Añadir punto actual
    fig.add_trace(go.Scatter(
        x=[len(historical)-1],
        y=[probability],
        mode='markers',
        marker=dict(size=12, color='red', symbol='star'),
        name='Valor Actual'
    ))
    
    fig.update_layout(
        title='Evolución de Probabilidad de Fallo',
        xaxis_title='Muestra (Horas)',
        yaxis_title='Probabilidad de Fallo',
        yaxis_range=[0, 1],
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Importancia de Características")
    
    if feature_importance is not None and len(feature_importance) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = feature_importance.head(10)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))[::-1]
        ax.barh(top_features['feature'], top_features['importance'], color=colors)
        ax.set_xlabel('Importancia')
        ax.set_title('Top 10 Características Más Importantes')
        ax.invert_yaxis()
        
        # Añadir valores
        for i, (idx, row) in enumerate(top_features.iterrows()):
            ax.text(row['importance'] + 0.01, i, f'{row["importance"]:.3f}', 
                   va='center', fontweight='bold')
        
        st.pyplot(fig)
    else:
        st.warning("⚠️ Datos de importancia de características no disponibles.")

# ============================================================================
# PARÁMETROS OPERATIVOS
# ============================================================================
st.markdown("---")
st.markdown("## 📋 Detalles Operativos")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔧 Parámetros de la Máquina")
    params_df = pd.DataFrame({
        'Parámetro': ['Temperatura Aire', 'Temperatura Proceso', 'Velocidad Rotación', 'Torque', 'Desgaste Herramienta'],
        'Valor': [air_temp, process_temp, rotational_speed, torque, tool_wear],
        'Unidad': ['K', 'K', 'RPM', 'Nm', 'min']
    })
    st.dataframe(params_df, hide_index=True, use_container_width=True)

with col2:
    st.markdown("#### 📊 Métricas Derivadas")
    temp_diff = process_temp - air_temp
    power = torque * (rotational_speed * 2 * np.pi / 60)
    thermal_eff = (process_temp / air_temp) * 100
    wear_torque = tool_wear * torque
    
    derived_df = pd.DataFrame({
        'Métrica': ['Diferencia Temperatura', 'Potencia', 'Eficiencia Térmica', 'Interacción Desgaste×Torque'],
        'Valor': [temp_diff, power, thermal_eff, wear_torque],
        'Unidad': ['K', 'W', '%', 'Nm·min']
    })
    st.dataframe(derived_df, hide_index=True, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>🏭 Sistema de Mantenimiento Predictivo con Análisis de Causa Raíz</p>
    <p><small>Desarrollado por Luis Ahumada Sánchez usando Streamlit, XGBoost y SHAP</small></p>
</div>
""", unsafe_allow_html=True)