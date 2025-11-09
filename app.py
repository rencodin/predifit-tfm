import streamlit as st
import pandas as pd
import os
from PIL import Image
import gdown

# ==========================
# CONFIGURACIÓN GENERAL
# ==========================
st.set_page_config(page_title="Dashboard Entrenamiento", layout="wide")

# ==========================
# DESCARGA AUTOMÁTICA DESDE GOOGLE DRIVE
# ==========================
drive_url = "https://drive.google.com/1cJ6u-_mXbdMMPO-o2QqG4wU160YK1pP"
local_path = "data/registro_def_st.csv"

# Crear carpeta 'data/' si no existe
os.makedirs("data", exist_ok=True)

# Intentar descargar
try:
    if not os.path.exists(local_path):
        st.info("Descargando dataset desde Google Drive...")
        gdown.download(drive_url, local_path, quiet=False)
        st.success("Dataset descargado correctamente.")
except Exception as e:
    st.warning("No se pudo descargar el archivo automáticamente. Puedes subirlo manualmente abajo.")
    archivo_manual = st.file_uploader("Sube el archivo registro_def.csv", type=["csv"])
    if archivo_manual:
        with open(local_path, "wb") as f:
            f.write(archivo_manual.getbuffer())
        st.success("Archivo subido correctamente.")

# ==========================
# IMPORTAR FUNCIONES EXTERNAS
# ==========================
from scripts.etl import rellenar_y_agregar
from scripts.transformacion import transformar_dataset
from scripts.eda_analisis import (
    resumen_ejercicios, ejercicio_mas_largo, ejercicios_sin_peso, volumen_por_ejercicio,
    grafico_duracion_media, grafico_duracion_por_semana, grafico_repes_por_semana,
    grafico_aceleracion, boxplot_pitch_roll, histograma_repeticiones, histograma_peso
)
from scripts.predicciones import (
    prediccion1, prediccion2, prediccion3, prediccion4, prediccion5, prediccion6
)

# ==========================
# SIDEBAR: Navegación y filtros
# ==========================
st.sidebar.title("📊 Panel de control")
vista = st.sidebar.radio("Selecciona vista:", ["1️⃣ Carga de datos", "2️⃣ Análisis EDA", "3️⃣ Predicciones"])

# ==========================
# VISTA 1: CARGA DE DATOS
# ==========================
if vista == "1️⃣ Carga de datos":
    st.title("📥 Carga de datos del reloj")

    st.subheader("📂 Opción 1: Subir y procesar archivo manualmente")
    archivo = st.file_uploader("Sube tu archivo CSV del reloj", type=["csv"])
    if archivo:
        st.success("Archivo cargado correctamente")
        st.dataframe(pd.read_csv(archivo).head())

        peso = st.number_input("Peso utilizado (kg)", min_value=0)
        serie = st.number_input("Series", min_value=1)
        repeticiones = st.number_input("Repeticiones", min_value=1)
        ejercicio = st.text_input("Ejercicio realizado")
        id_ejercicio = st.number_input("ID del ejercicio", min_value=1)
        semana = st.number_input("Semana", min_value=1)
        rutina = st.text_input("Rutina")

        if st.button("Procesar y añadir al dataset"):
            df_procesado = rellenar_y_agregar(
                archivo_csv=archivo,
                dataset_csv="data/registro.csv",
                peso=peso,
                serie=serie,
                repeticiones=repeticiones,
                ejercicio=ejercicio,
                id_ejercicio=id_ejercicio,
                semana=semana,
                rutina=rutina
            )
            st.success("Datos añadidos correctamente")
            st.dataframe(df_procesado.head())

        if st.button("Transformar y guardar dataset limpio"):
            df_transformado = transformar_dataset()
            st.success("Transformación completada")
            st.dataframe(df_transformado.head())

    st.markdown("---")
    st.subheader("📄 Opción 2: Usar directamente el dataset descargado")

    if st.button("Mostrar dataset registro_def_st.csv"):
        try:
            df_directo = pd.read_csv("data/registro_def_st.csv")
            st.success("Dataset cargado correctamente.")
            st.dataframe(df_directo.head())
        except Exception as e:
            st.error(f"No se pudo cargar el archivo: {e}")

    st.stop()

# ==========================
# CARGAR DATOS PARA VISTA 2 Y 3
# ==========================
df = pd.read_csv("data/registro_def_st.csv")

# ==========================
# FILTROS COMUNES
# ==========================
st.sidebar.header("Filtros")

ejercicio = st.sidebar.selectbox("Ejercicio:", options=["Todos"] + sorted(df["ejercicio"].dropna().unique()), index=0)
grupo_muscular = st.sidebar.selectbox("Grupo muscular:", options=["Todos", "Torso", "Pierna", "Brazos", "Espalda", "Core"])
peso_externo = st.sidebar.radio("¿Con peso externo?", ["Todos", "Sí", "No"])
semana = st.sidebar.multiselect("Semana:", options=sorted(df["semana"].unique()))
sexo = st.sidebar.radio("Sexo:", ["Mujer", "Hombre"])

peso_min, peso_max = st.sidebar.slider("Peso (kg):", int(df["peso"].min()), int(df["peso"].max()), (0, int(df["peso"].max())))
serie_min, serie_max = st.sidebar.slider("Series:", int(df["serie"].min()), int(df["serie"].max()), (1, int(df["serie"].max())))
repe_min, repe_max = st.sidebar.slider("Repeticiones:", int(df["repeticiones"].min()), int(df["repeticiones"].max()), (1, int(df["repeticiones"].max())))

df_filtrado = df.copy()
if ejercicio != "Todos":
    df_filtrado = df_filtrado[df_filtrado["ejercicio"] == ejercicio]
if grupo_muscular != "Todos":
    df_filtrado = df_filtrado[df_filtrado["grupo_muscular"] == grupo_muscular]
if peso_externo != "Todos":
    df_filtrado = df_filtrado[df_filtrado["peso"] > 0] if peso_externo == "Sí" else df_filtrado[df_filtrado["peso"] == 0]
if semana:
    df_filtrado = df_filtrado[df_filtrado["semana"].isin(semana)]
df_filtrado = df_filtrado[
    (df_filtrado["peso"].between(peso_min, peso_max)) &
    (df_filtrado["serie"].between(serie_min, serie_max)) &
    (df_filtrado["repeticiones"].between(repe_min, repe_max))
]

# ==========================
# VISTA 2: ANÁLISIS EDA
# ==========================
if vista == "2️⃣ Análisis EDA":
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### 🧍 Vista corporal por ejercicio")
        if ejercicio != "Todos":
            ejercicio_archivo = ejercicio.lower().replace(" ", "_").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
            carpeta_sexo = "female" if sexo == "Mujer" else "male"

            col_img1, col_img2 = st.columns(2)
            ruta_frontal = f"assets/{carpeta_sexo}/{ejercicio_archivo}_f.png"
            ruta_trasera = f"assets/{carpeta_sexo}/{ejercicio_archivo}_t.png"
            ruta_simple = f"assets/{carpeta_sexo}/{ejercicio_archivo}.png"

            mostrado = False
            if os.path.exists(ruta_frontal):
                col_img1.image(Image.open(ruta_frontal), caption=f"{ejercicio} (frontal)")
                mostrado = True
            if os.path.exists(ruta_trasera):
                col_img2.image(Image.open(ruta_trasera), caption=f"{ejercicio} (trasera)")
                mostrado = True
            if not mostrado and os.path.exists(ruta_simple):
                st.image(Image.open(ruta_simple), caption=f"{ejercicio}")
                mostrado = True
            if not mostrado:
                st.info("No hay imagen disponible para este ejercicio.")
        else:
            st.info("Selecciona un ejercicio para mostrar la zona entrenada.")

    with col2:
        st.markdown("## 📈 Dashboard EDA")
        resumen_ejercicios(df_filtrado)
        ejercicio_mas_largo(df_filtrado)
        ejercicios_sin_peso(df_filtrado)
        volumen_por_ejercicio(df_filtrado)
        grafico_duracion_media(df_filtrado)
        grafico_duracion_por_semana(df_filtrado)
        grafico_repes_por_semana(df_filtrado)
        grafico_aceleracion(df_filtrado)
        boxplot_pitch_roll(df_filtrado)
        histograma_repeticiones(df_filtrado)
        histograma_peso(df_filtrado)

    st.stop()

# ==========================
# VISTA 3: PREDICCIONES
# ==========================
if vista == "3️⃣ Predicciones":
    st.title("🔮 Predicciones de entrenamiento")

    filtro_pred = st.sidebar.selectbox("Filtro específico:", ["Todos", "Alta carga", "Baja carga"])

    col1, col2 = st.columns(2)
    with col1:
        prediccion1(df_filtrado)
        prediccion2(df_filtrado)
        prediccion3(df_filtrado)
    with col2:
        prediccion4(df_filtrado)
        prediccion5(df_filtrado)
        prediccion6(df_filtrado)


