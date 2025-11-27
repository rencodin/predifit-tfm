import streamlit as st
import pandas as pd
import os
from PIL import Image

# ==========================
# CONFIGURACIÓN GENERAL
# ==========================
st.set_page_config(page_title="Predifit - Cuadro de Entrenamiento", layout="wide")

# ==========================
# BASE DIR PARA RUTAS SEGURAS
# ==========================
BASE_DIR = os.path.dirname(__file__)

# ==========================
# IMÁGENES ZONAS ENTRENADAS CUERPO HUMANO
# ==========================
ejercicio_imagen = {
    "Curl de Bíceps (Mancuerna)": "curl_biceps",
    "Elevaciones laterales (Mancuerna)": "elev_lat",
    "Encogimiento de Hombros (Mancuerna)": "encog",
    "Flexión Declinada": "flex_dcl",
    "Impulso de Cadera a una Pierna (Mancuerna)": "impulso_cadera",
    "Press de banca (Mancuerna)": "press_banca",
    "Remo inclinado (Mancuerna)": "remo_inclinado",
    "Press de Hombros (Mancuerna)": "press_hombro",
    "Flexiones": "flexiones",
    "Sentadilla Búlgara": "sentadilla_bulgara",
    "Peso Muerto Rumano (Mancuerna)": "peso_muerto",
    "Sentadilla Goblet": "goblet",
    "Elevación de Gemelos de Pie (Mancuerna)": "gemelos",
    "Remo invertido": "remo_inv",
    "Extensión de Tríceps (Mancuerna)": "triceps",
    "Zancada (Mancuerna)": "zancada",
}

# ==========================
# IMÁGENES ZONAS ENTRENADAS EJERCICIO
# ==========================
ejercicio_listado_imagen = {
    "Curl de Bíceps (Mancuerna)": "biceps",
    "Elevaciones laterales (Mancuerna)": "elev_lat",
    "Encogimiento de Hombros (Mancuerna)": "encog",
    "Flexión Declinada": "flex_decl",
    "Impulso de Cadera a una Pierna (Mancuerna)": "impulso",
    "Press de banca (Mancuerna)": "pecho",
    "Remo inclinado (Mancuerna)": "remo",
    "Press de Hombros (Mancuerna)": "hombros",
    "Flexiones": "flexiones",
    "Sentadilla Búlgara": "bulgarian",
    "Peso Muerto Rumano (Mancuerna)": "peso_muerto",
    "Sentadilla Goblet": "goblet",
    "Elevación de Gemelos de Pie (Mancuerna)": "gemelos",
    "Remo invertido": "remo_inv",
    "Extensión de Tríceps (Mancuerna)": "triceps",
    "Zancada (Mancuerna)": "zancada",
}

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
# FUNCIONES PARA GOOGLE DRIVE
# ==========================
@st.cache_data
def cargar_csv_drive(file_id: str):
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    return pd.read_csv(url)

# IDs de tus archivos en Google Drive (cámbialos por los tuyos)
FILE_ID_REGISTRO = "1mnY7l-2eOWwnF-4b7GaVzh5dWiAI4RYR"
FILE_ID_REGISTRO_DEF = "1DOmzXX6snvE7ccHIFQk-QhYlnaohWQLo"

# ==========================
# SIDEBAR: Navegación y filtros
# ==========================
st.sidebar.image(os.path.join(BASE_DIR, "assets", "icono_claro.png"), use_container_width=True)
st.sidebar.title("📊 Panel de control")
vista = st.sidebar.radio("Selecciona vista:", ["1️⃣ Carga de datos", "2️⃣ Análisis EDA", "3️⃣ Predicciones"])

# ==========================
# VISTA 1: CARGA DE DATOS
# ==========================
if vista == "1️⃣ Carga de datos":
    st.image(os.path.join(BASE_DIR, "assets", "predifit_claro.png"), width=280)
    st.title("📥 Carga de datos del reloj")
    st.markdown("Página de carga de datos generados por el reloj. ")

    st.subheader("📂 Opción 1: Usar registro.csv y transformarlo")
    if st.button("Transformar registro.csv"):
        try:
            # Descargar y cargar desde Google Drive
            df_registro = cargar_csv_drive(FILE_ID_REGISTRO)
            # Pasar el dataframe a tu función de transformación
            df_transformado = transformar_dataset()  # si tu función ya lee registro.csv internamente, puedes adaptarla para recibir df_registro
            st.success("Transformación completada")
            st.dataframe(df_transformado.head())
        except Exception as e:
            st.error(f"No se pudo transformar el archivo: {e}")

    st.markdown("---")
    st.subheader("📄 Opción 2: Usar directamente registro_def_st.csv")
    if st.button("Mostrar registro_def_st.csv sin modificar"):
        try:
            df_directo = cargar_csv_drive(FILE_ID_REGISTRO_DEF)
            st.success("Dataset cargado correctamente desde Google Drive.")
            st.dataframe(df_directo.head())
        except Exception as e:
            st.error(f"No se pudo cargar el archivo desde Google Drive: {e}")

    st.stop()

# ==========================
# CARGAR DATOS PARA VISTA 2 Y 3
# ==========================
try:
    df = cargar_csv_drive(FILE_ID_REGISTRO_DEF)
    columnas_necesarias = {"ejercicio", "rutina", "peso", "serie", "repeticiones", "semana"}
    if not columnas_necesarias.issubset(df.columns):
        st.error("❌ El archivo cargado no contiene todas las columnas necesarias.")
        st.write("Columnas encontradas:", df.columns.tolist())
        st.stop()
except Exception as e:
    st.error(f"No se pudo cargar el archivo desde Google Drive: {e}")
    st.stop()

# ==========================
# FILTROS COMUNES
# ==========================
st.sidebar.header("Filtros")

ejercicio = st.sidebar.selectbox("Ejercicio:", options=["Todos"] + sorted(df["ejercicio"].dropna().unique()), index=0)

# Mostrar imagen justo debajo del filtro
if ejercicio != "Todos":
    archivo = ejercicio_listado_imagen.get(ejercicio)
    if archivo:
        ruta_ejecutado = os.path.join(BASE_DIR, "assets", "listado_ejercicios", f"{archivo}.png")
        if os.path.exists(ruta_ejecutado):
            st.sidebar.image(Image.open(ruta_ejecutado), caption=f"{ejercicio}", width=220)
        else:
            st.sidebar.info("No hay imagen disponible para este ejercicio.")
    else:
        st.sidebar.warning("Este ejercicio no está mapeado en listado_ejercicios.")

rutina = st.sidebar.selectbox("Rutina:", options=["Todos", "Tren Superior", "Tren Inferior"])
peso_externo = st.sidebar.radio("¿Con peso externo?", ["Todos", "Sí", "No"])
semana = st.sidebar.multiselect("Semana:", options=sorted(df["semana"].unique()))
sexo = st.sidebar.radio("Sexo:", ["Mujer", "Hombre"])


peso_min, peso_max = st.sidebar.slider("Peso (kg):", int(df["peso"].min()), int(df["peso"].max()), (0, int(df["peso"].max())))
serie_min, serie_max = st.sidebar.slider("Series:", int(df["serie"].min()), int(df["serie"].max()), (1, int(df["serie"].max())))
repe_min, repe_max = st.sidebar.slider("Repeticiones:", int(df["repeticiones"].min()), int(df["repeticiones"].max()), (1, int(df["repeticiones"].max())))

# Aplicar filtros
df_filtrado = df.copy()
if ejercicio != "Todos":
    df_filtrado = df_filtrado[df_filtrado["ejercicio"] == ejercicio]
if rutina != "Todos":
    df_filtrado = df_filtrado[df_filtrado["rutina"] == rutina]
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
    # Logo y título
    st.image(os.path.join(BASE_DIR, "assets", "predifit_claro.png"), width=280)
    st.title("📈 Análisis Exploratorio de Datos de Entrenamiento")

    # Slider para ajustar tamaño de imagen
    tamano_imagen = 190

    # Parte superior: tres columnas paralelas
    col_img1, col_img2, col_text = st.columns([1, 0.9, 1.1])

    carpeta_sexo = "female" if sexo == "Mujer" else "male"

    # Mostrar imágenes en col_img1 y col_img2
    if ejercicio != "Todos":
        archivo = ejercicio_imagen.get(ejercicio)

        if archivo:
            ruta_frontal = os.path.join(BASE_DIR, "assets", carpeta_sexo, f"{archivo}_f.png")
            ruta_trasera = os.path.join(BASE_DIR, "assets", carpeta_sexo, f"{archivo}_t.png")
            ruta_simple = os.path.join(BASE_DIR, "assets", carpeta_sexo, f"{archivo}.png")

            mostrado = False

            if os.path.exists(ruta_frontal):
                col_img1.image(Image.open(ruta_frontal), caption=f"{ejercicio} (frontal)", width=tamano_imagen)
                mostrado = True
            if os.path.exists(ruta_trasera):
                col_img2.image(Image.open(ruta_trasera), caption=f"{ejercicio} (trasera)", width=tamano_imagen)
                mostrado = True
            if not mostrado and os.path.exists(ruta_simple):
                col_img1.image(Image.open(ruta_simple), caption=f"{ejercicio}", width=tamano_imagen)
                mostrado = True
            if not mostrado:
                col_img1.info("No hay imagen disponible para este ejercicio.")
        else:
            col_img1.warning("Este ejercicio no está mapeado en el diccionario de imágenes.")
    else:
        ruta_entero_f = os.path.join(BASE_DIR, "assets", carpeta_sexo, "entero_f.png")
        ruta_entero_t = os.path.join(BASE_DIR, "assets", carpeta_sexo, "entero_t.png")

        if os.path.exists(ruta_entero_f):
            col_img1.image(Image.open(ruta_entero_f), caption="Cuerpo completo (frontal)", width=tamano_imagen)
        if os.path.exists(ruta_entero_t):
            col_img2.image(Image.open(ruta_entero_t), caption="Cuerpo completo (trasera)", width=tamano_imagen)

    # Caja de texto en la tercera columna
    col_text.markdown("### 📝 Notas")
    col_text.text_area("Escribe aquí tus observaciones:", 
        "Este análisis muestra la distribución de ejercicios realizados, su duración media y la carga aplicada por semana. "
        "Puedes añadir comentarios sobre patrones observados, ejercicios destacados o posibles ajustes en la rutina.")
    
    # Separador visual
    st.markdown("---")

    # Parte inferior: análisis ocupa todo el ancho
    st.subheader("📊 Resultados del análisis EDA")

    resumen_ejercicios(df_filtrado)
    ejercicio_mas_largo(df_filtrado)
    ejercicios_sin_peso(df_filtrado)
    volumen_por_ejercicio(df_filtrado)
    grafico_duracion_media(df_filtrado)
    grafico_duracion_por_semana(df_filtrado, ejercicio)
    grafico_repes_por_semana(df_filtrado, ejercicio)
    grafico_aceleracion(df_filtrado, ejercicio)
    boxplot_pitch_roll(df_filtrado)
    histograma_repeticiones(df_filtrado, ejercicio)
    histograma_peso(df_filtrado, ejercicio)

    st.stop()


# ==========================
# VISTA 3: PREDICCIONES
# ==========================
if vista == "3️⃣ Predicciones":
    st.image(os.path.join(BASE_DIR, "assets", "predifit_claro.png"), width=280)
    st.title("🔮 Predicciones basadas en tu entrenamiento")
    st.markdown("Página de predicciones")

    # Aquí podrías añadir filtros específicos si lo deseas
    # filtro_pred = st.sidebar.selectbox("Filtro específico:", ["Todos", "Alta carga", "Baja carga"])

    prediccion1(df_filtrado)
    prediccion2(df_filtrado)
    prediccion3(df_filtrado)
    prediccion4(df_filtrado)
    prediccion5(df_filtrado)
    prediccion6(df_filtrado)





