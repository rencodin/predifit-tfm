import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -----------------------------
# Diccionario ejercicio → id
# -----------------------------
def obtener_diccionario_ejercicios(df):
    if not {"ejercicio","id_ejercicio"}.issubset(df.columns):
        return {}
    return df.dropna(subset=["ejercicio", "id_ejercicio"])\
             .drop_duplicates(subset=["ejercicio"])\
             .set_index("ejercicio")["id_ejercicio"].to_dict()

# -----------------------------
# Cache de entrenamiento modelos
# -----------------------------
@st.cache_data
def entrenar_xgb(df):
    if not {"peso","semana","id_ejercicio","repeticiones"}.issubset(df.columns):
        return None, {}
    df = df.dropna(subset=["peso", "semana", "id_ejercicio", "repeticiones"])
    X = df[["semana", "id_ejercicio", "repeticiones"]]
    y = df["peso"]
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, obtener_diccionario_ejercicios(df)

@st.cache_data
def entrenar_lr(df):
    if not {"id_ejercicio","peso","serie","repeticiones","semana","duracion_media"}.issubset(df.columns):
        return None, {}
    X = df[["id_ejercicio", "peso", "serie", "repeticiones", "semana"]]
    y = df["duracion_media"]
    model = LinearRegression()
    model.fit(X, y)
    return model, obtener_diccionario_ejercicios(df)

# -----------------------------
# 1. Predicción de carga (peso)
# -----------------------------
def prediccion1(df):
    st.subheader("📌 Predicción de carga (peso) con XGBoost")
    model, ejercicios_dict = entrenar_xgb(df)
    if model is None:
        st.warning("No hay columnas suficientes para entrenar el modelo de carga.")
        return

    semana = st.number_input("Semana", min_value=1, max_value=52, value=10)
    ejercicio_id = st.selectbox("ID del ejercicio", sorted(df["id_ejercicio"].dropna().unique()))
    repeticiones = st.number_input("Repeticiones", min_value=1, max_value=50, value=12)

    pred = model.predict([[semana, ejercicio_id, repeticiones]])[0]
    st.metric(label="Peso estimado", value=f"{pred:.2f} kg")

    st.markdown("""
    🔍 **¿Cómo interpretar esta predicción?**

    - El aumento de fuerza en el levantamiento de pesas es un proceso gradual que se logra mediante una adecuada progresión. Con una planificación correcta, una técnica bien aplicada y el descanso necesario, los músculos podrán crecer y recuperarse en el tiempo que requieren.
    - La variación en el entrenamiento —ya sea a través de cambios en los ejercicios, el número de repeticiones o las series realizadas— puede estimular nuevos avances y prevenir el estancamiento.
    - La progresión que sigas será de tipo lineal, y el peso considerado en esta predicción se basa en los datos históricos que has registrado en semanas anteriores. Es importante señalar que esta información no sustituye en ningún caso la orientación de un profesional del entrenamiento.

    📌 *Este modelo no sustituye la supervisión profesional.*
    """)

# -----------------------------
# 2. Predicción de duración media
# -----------------------------
def prediccion2(df):
    st.subheader("⏱️ Predicción de duración media con regresión")
    model, ejercicios_dict = entrenar_lr(df)
    if model is None:
        st.warning("No hay columnas suficientes para entrenar el modelo de duración.")
        return

    ejercicio_nombre = st.selectbox("Ejercicio", sorted(ejercicios_dict.keys()), key="nombre_ejercicio_pred2")
    ejercicio_id = ejercicios_dict[ejercicio_nombre]

    peso = st.number_input("Peso (kg)", min_value=0.0, max_value=200.0, value=30.0, key="peso_pred2")
    serie = st.number_input("Serie", min_value=1, max_value=10, value=4, key="serie_pred2")
    repeticiones = st.number_input("Repeticiones", min_value=1, max_value=50, value=12, key="repes_pred2")
    semana = st.number_input("Semana", min_value=1, max_value=52, value=10, key="semana_pred2")

    entrada = [[ejercicio_id, peso, serie, repeticiones, semana]]
    pred = model.predict(entrada)[0]
    st.metric(label="Duración estimada", value=f"{pred:.2f} segundos")

    st.markdown("""
    🔍 **¿Cómo interpretar esta predicción?**

    - La duración de cada sesión de levantamiento de pesas influye directamente en la calidad del progreso. No se trata solo de cuánto peso se levanta, sino de cuánto tiempo se mantiene el esfuerzo y cómo se distribuyen las pausas.
    - Un entrenamiento demasiado corto puede no generar el estímulo suficiente, mientras que uno excesivamente largo puede provocar fatiga acumulada y disminuir la capacidad de recuperación. Encontrar un equilibrio entre tiempo bajo tensión y descanso es esencial para que los músculos asimilen el trabajo realizado.
    - Además, ajustar la duración de los ejercicios según el objetivo —ya sea fuerza máxima, hipertrofia o resistencia muscular— permite orientar mejor los resultados. Por ejemplo, sesiones más breves e intensas favorecen la fuerza, mientras que entrenamientos más prolongados con cargas moderadas estimulan la resistencia.
    - Las predicciones que se realicen sobre la duración del ejercicio se basan en registros históricos de tus entrenamientos previos. Sin embargo, estas estimaciones son solo una referencia y nunca deben reemplazar la planificación personalizada que puede ofrecer un profesional del entrenamiento.
                
    📌 *Este modelo no sustituye la supervisión profesional.*
    """)

# -----------------------------
# 3. Clasificación de fallo técnico
# -----------------------------
def prediccion3(df):
    st.subheader("⚠️ Clasificación de fallo técnico por rotación")

    if not {"pitch_grados","roll_grados","yaw_grados"}.issubset(df.columns):
        st.warning("No hay datos de rotación angular disponibles.")
        return

    df_std = df.groupby(["id_ejercicio","serie","repeticiones","semana","peso"]).agg({
        "pitch_grados":"std","roll_grados":"std","yaw_grados":"std"
    }).reset_index()

    # 🔧 Eliminar filas con NaN
    df_std = df_std.dropna(subset=["pitch_grados","roll_grados","yaw_grados"])

    df_std["fallo_tecnico"] = (
        (df_std["pitch_grados"] > 15) |
        (df_std["roll_grados"] > 15) |
        (df_std["yaw_grados"] > 15)
    ).astype(int)

    X = df_std[["pitch_grados","roll_grados","yaw_grados"]]
    y = df_std["fallo_tecnico"]

    if y.nunique() < 2:
        st.warning("No hay suficiente variación para entrenar el modelo de fallo técnico.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test,y_pred)
    st.write("Matriz de confusión:")
    st.write(cm)
    st.text(classification_report(y_test,y_pred))

    
    st.markdown("""
    🔍 **¿Cómo interpretar esta predicción?**
                
    - Analizar si la muñeca rota o se inclina más de lo esperado (roll, yaw) puede ayudar a detectar “ruido” en la técnica al llegar al fallo. 
      Esto puede llevar a la conclusión de que aparecen patrones de mala técnica con la fatiga.
    - Pitch, roll y yaw son términos del inglés que hacen referencia a 
      A continuación, se muestra el rango de movimientos en grados empleado: 
        • Pitch: Inclinación hacia adelante/atrás --> -90° a +90 
        • Roll: Inclinación lateral --> -90° a +90° 
        • Yaw: Rotación horizontal --> 0° a 360° 
    - El modelo analiza la variación angular (pitch, roll, yaw) para detectar fallos técnicos.
    - Si la variación es alta, puede indicar asimetrías o desviaciones en la ejecución.
    - Si la variación es baja, refleja estabilidad y control en el movimiento.

    📌 *Este modelo no sustituye la supervisión profesional.*
    """)

# -----------------------------
# 4. PCA de rotaciones
# -----------------------------
def prediccion4(df):
    st.subheader("📐 PCA de rotaciones angulares")
    if not {"pitch_grados","roll_grados","yaw_grados"}.issubset(df.columns):
        st.warning("No hay datos de rotación angular disponibles.")
        return

    X = df[["pitch_grados","roll_grados","yaw_grados"]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(X_pca[:,0],X_pca[:,1],alpha=0.6)
    ax.set_title("PCA de rotaciones")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    st.pyplot(fig)

    st.markdown("""
    🔍 **Interpretación para el usuario**

    - La primera componente (PC1) explica aproximadamente **{var_pc1:.0%}** de la variación en tus rotaciones.
      Esto suele reflejar diferencias dominantes, como la **velocidad de ejecución** o la magnitud del movimiento.
    - La segunda componente (PC2) explica alrededor de **{var_pc2:.0%}**, capturando variaciones adicionales,
      como la **simetría o asimetría** en la técnica.
    - En conjunto, estas dos componentes resumen más del **{(var_pc1+var_pc2):.0%}** de la información original,
      lo que permite visualizar patrones complejos en un plano bidimensional.

    👉 Si tus repeticiones aparecen agrupadas en el gráfico, significa que tu técnica es **consistente**.
    👉 Si ves puntos alejados del grupo, pueden indicar **fallos técnicos o desviaciones angulares** 
    (pitch, roll o yaw fuera de rango).

    📌 *Este análisis es una herramienta de apoyo para detectar patrones, no sustituye la supervisión profesional.*
    """)

# -----------------------------
# 5. Clustering de series
# -----------------------------
def prediccion5(df):
    st.subheader("🧠 Clustering de series (K-Means + PCA)")
    if not {"duracion_media","volumen_total"}.issubset(df.columns):
        st.warning("No hay columnas suficientes para clustering.")
        return

    df = df.copy()
    df["velocidad"] = 1 / df["duracion_media"]
    features = ["duracion_media","velocidad","volumen_total"]
    X = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3,random_state=42,n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    df_plot = pd.DataFrame(X_pca,columns=["PCA 1","PCA 2"])
    df_plot["cluster"] = df["cluster"]

    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(x="PCA 1",y="PCA 2",hue="cluster",data=df_plot,palette="tab10",s=60)
    ax.scatter(centroids_pca[:,0],centroids_pca[:,1],marker="*",s=250,color="black",label="Centros")
    ax.set_title("Clusters K-Means de series")
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    🔍 **Interpretación para el usuario**

    El análisis agrupa tus series en tres patrones principales:

    - **Cluster 0**: Series rápidas, con baja duración media y alta velocidad. 
      👉 Suelen reflejar ejecuciones explosivas o de fuerza máxima.
    - **Cluster 1**: Series más lentas y controladas, con mayor duración media. 
      👉 Asociadas a trabajo de hipertrofia o resistencia muscular.
    - **Cluster 2**: Series con mayor volumen total y tendencia a la fatiga. 
      👉 Aquí se observa acumulación de esfuerzo, donde la técnica puede variar.

    📌 *Este agrupamiento te ayuda a identificar cómo varía tu ejecución entre rapidez, control y fatiga. 
    Si ves que predominan las series del cluster de fatiga, puede ser útil ajustar descansos o cargas.*
    """)

# -----------------------------
# 6. Histograma de carga estimada por semana
# -----------------------------
def prediccion6(df):
    st.subheader("📊 Histograma de carga estimada por semana")

    model, ejercicios_dict = entrenar_xgb(df)
    if model is None:
        st.warning("No hay columnas suficientes para entrenar el modelo de carga.")
        return

    if not ejercicios_dict:
        st.warning("No se encontraron ejercicios con ID para realizar la predicción.")
        return

    ejercicio_nombre = st.selectbox("Ejercicio", sorted(ejercicios_dict.keys()), key="nombre_ejercicio_pred6")
    ejercicio_id = ejercicios_dict[ejercicio_nombre]

    repeticiones = st.number_input("Repeticiones", min_value=1, max_value=50, value=12, key="repes_pred6")

    semanas = sorted(df["semana"].dropna().unique())
    if not semanas.any():
        st.warning("No hay semanas disponibles en el dataset.")
        return

    predicciones = [model.predict([[s, ejercicio_id, repeticiones]])[0] for s in semanas]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(semanas, predicciones, color="#53A0E7")
    ax.set_title(f"Carga estimada por semana (Ejercicio: {ejercicio_nombre}, {repeticiones} repes)")
    ax.set_xlabel("Semana")
    ax.set_ylabel("Peso estimado (kg)")
    st.pyplot(fig)

    st.markdown(f"""
    🔍 **Interpretación para el usuario**

    - El histograma muestra cómo evoluciona la **carga estimada (peso en kg)** para el ejercicio seleccionado
      a lo largo de las semanas, considerando un número fijo de repeticiones (**{repeticiones}**).
    - Si observas un **incremento progresivo**, significa que tu entrenamiento está generando adaptaciones
      y mejoras en la fuerza.
    - Si la carga se mantiene estable o disminuye, puede reflejar un **estancamiento** o necesidad de ajustar
      la planificación (variar repeticiones, series o descansos).
    - Este análisis se basa en tus datos históricos y ofrece una referencia objetiva para seguir tu progreso.

    📌 *Recuerda: estas predicciones son una guía, pero no sustituyen la supervisión profesional.*
    """)

