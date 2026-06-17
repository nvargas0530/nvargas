# EDA — Bomba de Agua Industrial 💧⚙️

Dashboard interactivo de Análisis Exploratorio de Datos para el dataset **Pump Sensor Data**, que contiene lecturas de 52 sensores instalados en una bomba de agua industrial monitoreada durante 5 meses (Abril–Agosto 2018).

## 📊 Contenido del Dashboard

| Tab | Descripción |
|-----|-------------|
| 📊 Overview | Métricas generales, distribución de estados y valores faltantes |
| 📈 Series de Tiempo | Evolución temporal de sensores y estado operativo |
| 🔍 Distribuciones | Estadística descriptiva, histogramas y Q-Q plots |
| 🔗 Correlaciones | Mapa de calor y top pares correlacionados |
| ⚠️ Anomalías | Boxplots por estado y detección de outliers (IQR) |
| 📋 Inferencia | Prueba de Kruskal-Wallis y violin plots comparativos |

## 🚀 Ejecutar localmente

```bash
pip install -r requirements.txt
python app.py
```

## 🌐 Deploy en Railway

1. Conecta este repositorio en [railway.app](https://railway.app)
2. Railway detecta automáticamente el `Procfile`
3. ¡Listo!

## 📁 Datos

Dataset original: [Pump Sensor Data — Kaggle](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)
