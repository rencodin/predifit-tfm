import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, median_absolute_error,
    r2_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Predicción de carga (peso) con XGBoost
def prediccion1(df):
    st.subheader("📌 Predicción de carga (peso) con XGBoost")

    X = df[["semana", "id_ejercicio", "repeticiones"]]
    y = df["peso"]
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    semana = st.number_input("Semana", min_value=1, max_value=52, value=10)
    ejercicio_id = st.selectbox("ID del ejercicio", sorted(df["id_ejercicio"].unique()))
    repeticiones = st.number_input("Repeticiones", min_value=1, max_value=50, value=12)

    pred = model.predict([[semana, ejercicio_id, repeticiones]])[0]
    st.metric(label="Peso estimado", value=f"{pred:.2f} kg")

# 2. Predicción de duración media con regresión lineal
def prediccion2(df):
    st.subheader("⏱️ Predicción de duración media con regresión")

    X = df[["id_ejercicio", "peso", "serie", "repeticiones", "semana"]]
    y = df["duracion_media"]
    model = LinearRegression()
    model.fit(X, y)

    ejercicio_id = st.selectbox("ID del ejercicio", sorted(df["id_ejercicio"].unique()), key="duracion_ej")
    peso = st.number_input("Peso (kg)", min_value=0.0, max_value=200.0, value=30.0)
    serie = st.number_input("Serie", min_value=1, max_value=10, value=4)
    repeticiones = st.number_input("Repeticiones", min_value=1, max_value=50, value=12)
    semana = st.number_input("Semana", min_value=1, max_value=52, value=10)

    pred = model.predict([[ejercicio_id, peso, serie, repeticiones, semana]])[0]
    st.metric(label="Duración estimada", value=f"{pred:.2f} segundos")

# 3. Clasificación de fallo técnico por rotación angular
def prediccion3(df):
    st.subheader("⚠️ Clasificación de fallo técnico por rotación")

    df_std = df.groupby(["id_ejercicio", "serie", "repeticiones", "semana", "peso"]).agg({
        "pitch_grados": "std", "roll_grados": "std", "yaw_grados": "std"
    }).reset_index()

    df_std["fallo_tecnico"] = (
        (df_std["pitch_grados"] > 15) |
        (df_std["roll_grados"] > 15) |
        (df_std["yaw_grados"] > 15)
    ).astype(int)

    X = df_std[["pitch_grados", "roll_grados", "yaw_grados"]]
    y = df_std["fallo_tecnico"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("Matriz de confusión:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Reporte de clasificación:")
    st.text(classification_report(y_test, y_pred))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(x="fallo_tecnico", y="pitch_grados", data=df_std, ax=axes[0])
    sns.boxplot(x="fallo_tecnico", y="roll_grados", data=df_std, ax=axes[1])
    sns.boxplot(x="fallo_tecnico", y="yaw_grados", data=df_std, ax=axes[2])
    axes[0].set_title("Pitch vs Fallo Técnico")
    axes[1].set_title("Roll vs Fallo Técnico")
    axes[2].set_title("Yaw vs Fallo Técnico")
    st.pyplot(fig)

# 4. PCA de rotaciones
def prediccion4(df):
    st.subheader("📐 PCA de rotaciones angulares")

    X = df[["pitch_grados", "roll_grados", "yaw_grados"]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    st.write(f"Varianza explicada (PC1): {pca.explained_variance_ratio_[0]:.2f}")
    st.write(f"Varianza explicada (PC2): {pca.explained_variance_ratio_[1]:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    ax.set_title("PCA de rotaciones")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    st.pyplot(fig)

# 5. Clustering de series por duración, velocidad y volumen
def prediccion5(df):
    st.subheader("🧠 Clustering de series (K-Means + PCA)")

    df = df.copy()
    df["velocidad"] = 1 / df["duracion_media"]
    features = ["duracion_media", "velocidad", "volumen_total"]
    X = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    df_plot = pd.DataFrame(X_pca, columns=["PCA 1", "PCA 2"])
    df_plot["cluster"] = df["cluster"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x="PCA 1", y="PCA 2", hue="cluster", data=df_plot, palette="tab10", s=60)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker="*", s=250, color="black", label="Centros")
    ax.set_title("Clusters K-Means de series")
    ax.legend()
    st.pyplot(fig)

# 6. Histograma de carga estimada por semana
def prediccion6(df):
    st.subheader("📊 Histograma de carga estimada por semana")

    X = df[["semana", "id_ejercicio", "repeticiones"]]
    y = df["peso"]
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    ejercicio_id = st.selectbox("ID del ejercicio", sorted(df["id_ejercicio"].unique()), key="hist_ej")
    repeticiones = st.number_input("Repeticiones", min_value=1, max_value=50, value=12)

    semanas = sorted(df["semana"].unique())
    predicciones = [model.predict([[s, ejercicio_id, repeticiones]])[0] for s in semanas]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(semanas, predicciones, color="#F2AB6D")
    ax.set_title(f"Carga estimada por semana (Ejercicio {ejercicio_id}, {repeticiones} repes)")
    ax.set_xlabel("Semana")
    ax.set_ylabel("Peso estimado (kg)")
    st.pyplot(fig)
