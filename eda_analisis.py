import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Conteo de ejercicios por tipo
def resumen_ejercicios(df):
    st.subheader("📋 Resumen de ejercicios")

    total = len(df["ejercicio"].unique())
    inferior = df.loc[df["rutina"].isin(["PA", "PB"]), "ejercicio"].unique()
    superior = df.loc[df["rutina"].isin(["TA", "TB"]), "ejercicio"].unique()

    st.markdown(f"- **Total de ejercicios:** {total}")
    st.markdown(f"- **Tren inferior:** {', '.join(inferior)} ({len(inferior)})")
    st.markdown(f"- **Tren superior:** {', '.join(superior)} ({len(superior)})")

# 2. Ejercicio con mayor duración
def ejercicio_mas_largo(df):
    st.subheader("⏱️ Ejercicio más largo")
    duraciones = df.groupby("ejercicio")["segundos"].mean()
    ejercicio_max = duraciones.idxmax()
    maximo = round(duraciones.max(), 2)
    st.markdown(f"- **Ejercicio más largo:** {ejercicio_max} con {maximo} segundos")

# 3. Ejercicios sin peso externo
def ejercicios_sin_peso(df):
    st.subheader("🚫 Ejercicios sin peso externo")
    no_peso = df.loc[df["peso"] == 0, "ejercicio"].unique()
    st.markdown(f"- **Ejercicios sin carga:** {', '.join(no_peso)}")

# 4. Ejercicio con mayor y menor volumen
def volumen_por_ejercicio(df):
    st.subheader("🏋️ Volumen por ejercicio")
    volumen = df.groupby("ejercicio")["peso"].count()
    ejercicio_max = volumen.idxmax()
    ejercicio_min = volumen.idxmin()
    st.markdown(f"- **Mayor volumen:** {ejercicio_max} ({volumen.max()} registros)")
    st.markdown(f"- **Menor volumen:** {ejercicio_min} ({volumen.min()} registros)")

# 5. Duración media por ejercicio (gráfico)
def grafico_duracion_media(df):
    st.subheader("📊 Duración media por ejercicio")
    media = df.groupby("ejercicio")["segundos"].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(media.index, media.values)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), round(bar.get_height(), 2),
                ha='center', va='bottom', fontsize=9)
    ax.set_title("Duración media por ejercicio")
    ax.set_ylabel("Segundos")
    ax.set_xticklabels(media.index, rotation=45, ha='right')
    st.pyplot(fig)

# 6. Duración por semana para ejercicio ID 7
def grafico_duracion_por_semana(df):
    st.subheader("📈 Duración semanal (ejercicio ID 7)")
    media = df[df["id_ejercicio"] == 7].groupby("semana")["segundos"].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(media.index, media.values)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), round(bar.get_height(), 2),
                ha='center', va='bottom', fontsize=9)
    ax.set_title("Duración media por semana")
    ax.set_ylabel("Segundos")
    st.pyplot(fig)

# 7. Repeticiones por semana para ejercicio ID 7
def grafico_repes_por_semana(df):
    st.subheader("🔁 Repeticiones por semana (ejercicio ID 7)")
    media = df[df["id_ejercicio"] == 7].groupby("semana")["repeticiones"].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(media.index, media.values)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), round(bar.get_height(), 2),
                ha='center', va='bottom', fontsize=9)
    ax.set_title("Repeticiones promedio por semana")
    ax.set_ylabel("Repeticiones")
    st.pyplot(fig)

# 8. Aceleración por semana (ejercicio ID 7)
def grafico_aceleracion(df):
    st.subheader("📉 Aceleración por semana (ejercicio ID 7)")
    semanas = range(1, 9)
    df_ejercicio = df[df["id_ejercicio"] == 7]
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True, constrained_layout=True)
    for i, semana in enumerate(semanas):
        ax = axs[i // 4, i % 4]
        df_semana = df_ejercicio[(df_ejercicio["semana"] == semana) & (df_ejercicio["segundos"] <= 30)]
        ax.plot(df_semana["segundos"], df_semana["accX"], label="Acc X", color="blue")
        ax.plot(df_semana["segundos"], df_semana["accY"], label="Acc Y", color="orange")
        ax.plot(df_semana["segundos"], df_semana["accZ"], label="Acc Z", color="green")
        ax.set_title(f"Semana {semana}")
        ax.set_xlabel("Segundos")
        ax.set_ylabel("Aceleración")
        ax.legend()
    st.pyplot(fig)

# 9. Boxplot de pitch y roll por semana
def boxplot_pitch_roll(df):
    st.subheader("📦 Distribución de pitch y roll por semana")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x="semana", y="pitch", data=df, ax=axs[0])
    axs[0].set_title("Pitch por semana")
    sns.boxplot(x="semana", y="roll", data=df, ax=axs[1])
    axs[1].set_title("Roll por semana")
    st.pyplot(fig)

# 10. Histograma de repeticiones (ID 7)
def histograma_repeticiones(df):
    st.subheader("📊 Histograma de repeticiones (ID 7)")
    repes = pd.to_numeric(df[df["id_ejercicio"] == 7]["repeticiones"], errors="coerce").dropna()
    if repes.empty:
        st.warning("No hay datos para ID 7")
        return
    bins = range(int(repes.min()), int(repes.max()) + 2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(repes, bins=bins, color="#F2AB6D", rwidth=0.85)
    ax.set_title("Histograma de repeticiones")
    ax.set_xlabel("Repeticiones")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

# 11. Histograma de peso (ID 7)
def histograma_peso(df):
    st.subheader("🏋️ Histograma de carga (ID 7)")
    pesos = pd.to_numeric(df[df["id_ejercicio"] == 7]["peso"], errors="coerce").dropna()
    if pesos.empty:
        st.warning("No hay datos para ID 7")
        return
    bins = range(int(pesos.min()), int(pesos.max()) + 2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pesos, bins=bins, color="#F2AB6D", rwidth=0.85)
    ax.set_title("Histograma de carga")
    ax.set_xlabel("Peso (kg)")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)
