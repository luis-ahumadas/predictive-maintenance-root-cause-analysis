# 🏭 Predictive Maintenance with Root Cause Analysis

## Descripción del Proyecto

Sistema de Machine Learning para predecir fallos en maquinaria industrial y diagnosticar su causa raíz, permitiendo intervenciones de mantenimiento más precisas y oportunas.

### 🎯 Objetivos
- Predecir fallos en máquinas de fresado con alta sensibilidad
- Identificar la causa específica de cada fallo (desgaste, sobreesfuerzo, térmico, etc.)
- Proporcionar explicaciones interpretables para decisiones de mantenimiento

### 📊 Dataset
- **AI4I 2020 Predictive Maintenance Dataset** (Kaggle)
- 10,000 registros de sensores industriales
- 5 modos de fallo etiquetados: TWF, HDF, PWF, OSF, RNF

### 🛠️ Metodología
1. **Ingeniería de características** basada en física (Potencia, Eficiencia Térmica)
2. **Modelado** con XGBoost y SMOTE para manejar datos desbalanceados
3. **Interpretación** con SHAP para análisis de causa raíz
4. **Dashboard** interactivo para monitoreo en tiempo real

### 📈 Resultados
- **Recall:** 92% (detección de fallos)
- **Precisión en diagnóstico de causa:** 78%
- **AUC-ROC:** 0.96

### 📂 Estructura del Proyecto
```bash
├── data/       # Datos procesados
├── notebooks/  # Análisis exploratorio y modelado
├── dashboard/  # Aplicación Streamlit
├── reports/    # Figuras y visualizaciones
└── models/     # Modelos entrenados
```

### 🚀 Cómo Ejecutar
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar dashboard
streamlit run dashboard/app.py
```
