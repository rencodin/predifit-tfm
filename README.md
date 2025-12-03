# Predifit - Web App de Análisis de Entrenamiento ![LogoPredifit](assets/predifit_oscuro.png)
## Trabajo Fin de Máster "Análisis Predictivo del Rendimiento en Entrenamiento de Fuerza mediante Machine Learning" - IMF Business School
## Autora: Renata Flores Hidalgo

Este repositorio contiene la implementación en Streamlit del proyecto “Análisis Predictivo del Rendimiento en Entrenamiento de Fuerza mediante Machine Learning”.

URL para acceder ! ---> https://predifit-tfm-4srjsmn48dse4nqtke5yfs.streamlit.app/ 

📑 Descripción

🏋️‍♀️ Predifit es una aplicación web diseñada para: 🏋️‍♀️

- Carga de datos desde Apple Watch: permite adjuntar registros de entrenamiento obtenidos desde el dispositivo.
- Interfaz interactiva: visualización y análisis de métricas de rendimiento en tiempo real.
- Dataset de prueba: incluye el conjunto de datos final utilizado en el TFM.
  
:exclamation:❗Atención:
  
⚠️ Recomendaciones de uso

💡 Mejor testeo en local: se recomienda descargar el contenido de la carpeta version_local para trabajar con datasets completos y obtener un rendimiento óptimo.

- Versión en GitHub: está optimizada para mostrar el estado del arte del proyecto, pero no incluye el dataset completo por limitaciones de la plataforma. Se filtran por las dos primeras semanas de entrenamiento.

- Archivo CSV grande: el fichero registro_def.csv ocupa aproximadamente 200 MB, por lo que es preferible trabajar en local para visualizar correctamente el Análisis Exploratorio de Datos (EDA) y las Predicciones.

- No usar la opción 1: en la primera vista "1️⃣ Carga de datos", la pestaña “Opción 1: Usar registro.csv y transformarlo” carga un dataset en LFS (no en CSV estándar). Aunque puede visualizarse en la carpeta data en formato raw, supera el límite permitido por GitHub.

❗ Limitaciones en la versión online:

- En las vistas “2️⃣ Análisis EDA” y “3️⃣ Predicciones” solo se muestran resultados parciales, ya que se trabaja con las dos primeras semanas del registro de entrenamiento.

- Los filtros aplicados en el menú lateral de la app modificarán dinámicamente los resultados del análisis, por lo que cada ejecución puede mostrar salidas distintas.

👉 Conclusión: para un análisis completo y fluido, se recomienda TRABAJAR en LOCAL con la carpeta version_local.

Versiones disponibles

- Versión sin modificar: dataset completo original -> "registro.csv"
- Versión final depurada: utilizada en el análisis predictivo del TFM -> "registro_def.csv"

📦 Tecnologías utilizadas

- Python 🐍
- Streamlit 🌐
- Machine Learning (XGBoost, regresión, etc.) 🤖
- Apple Watch, Integración con Sensor Log App 📊

🎯 Objetivo del proyecto

Facilitar la exploración de resultados y la validación de modelos predictivos en un entorno accesible, reproducible y orientado a profesionales del entrenamiento y la investigación.
