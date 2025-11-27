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
    if "segundos" not in df.columns or df.empty:
        st.warning("No hay datos de duración disponibles.")
        return
    duraciones = df.groupby("ejercicio")["segundos"].mean()
    ejercicio_max = duraciones.idxmax()
    maximo = round(duraciones.max(), 2)
    st.markdown(f"- **Ejercicio más largo:** {ejercicio_max} con {maximo} segundos")

# 3. Ejercicios sin peso externo
def ejercicios_sin_peso(df):
    st.subheader("🚫 Ejercicios sin peso externo")
    no_peso = df.loc[df["peso"] == 0, "ejercicio"].unique()
    if len(no_peso) == 0:
        st.info("No se encontraron ejercicios sin peso externo.")
    else:
        st.markdown(f"- **Ejercicios sin carga:** {', '.join(no_peso)}")

# 4. Ejercicio con mayor y menor volumen
def volumen_por_ejercicio(df):
    st.subheader("🏋️ Volumen por ejercicio")
    volumen = df.groupby("ejercicio")["peso"].count()
    if volumen.empty:
        st.warning("No hay datos de volumen disponibles.")
        return
    ejercicio_max = volumen.idxmax()
    ejercicio_min = volumen.idxmin()
    st.markdown(f"- **Mayor volumen:** {ejercicio_max} ({volumen.max()} registros)")
    st.markdown(f"- **Menor volumen:** {ejercicio_min} ({volumen.min()} registros)")

# 5. Duración media por ejercicio (gráfico)
def grafico_duracion_media(df):
    st.subheader("📊 Duración media por ejercicio")
    if "segundos" not in df.columns or df.empty:
        st.warning("No hay datos de duración disponibles.")
        return
    media = df.groupby("ejercicio")["segundos"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(media.index, media.values)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), round(bar.get_height(), 2),
                ha='center', va='bottom', fontsize=9)
    ax.set_title("Duración media por ejercicio")
    ax.set_ylabel("Segundos")
    ax.set_xticklabels(media.index, rotation=45, ha='right')
    st.pyplot(fig)

# 6. Duración por semana 
def grafico_duracion_por_semana(df, ejercicio=None):
    st.subheader("📈 Duración semanal")
    if "segundos" not in df.columns or df.empty:
        st.warning("No hay datos de duración disponibles.")
        return
    media = df.groupby("semana")["segundos"].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(media.index, media.values)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), round(bar.get_height(), 2),
                ha='center', va='bottom', fontsize=9)
    ax.set_title(f"Duración media por semana {f'({ejercicio})' if ejercicio and ejercicio!='Todos' else ''}")
    ax.set_ylabel("Segundos")
    st.pyplot(fig)

# 7. Repeticiones por semana 
def grafico_repes_por_semana(df, ejercicio=None):
    st.subheader("🔁 Repeticiones por semana")
    if "repeticiones" not in df.columns or df.empty:
        st.warning("No hay datos de repeticiones disponibles.")
        return
    media = df.groupby("semana")["repeticiones"].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(media.index, media.values)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), round(bar.get_height(), 2),
                ha='center', va='bottom', fontsize=9)
    ax.set_title(f"Repeticiones promedio por semana {f'({ejercicio})' if ejercicio and ejercicio!='Todos' else ''}")
    ax.set_ylabel("Repeticiones")
    st.pyplot(fig)

# 8. Aceleración por semana 
def grafico_aceleracion(df, ejercicio=None):
    st.subheader("📉 Aceleración por semana")
    if not {"accX","accY","accZ","segundos"}.issubset(df.columns):
        st.warning("No hay datos de aceleración disponibles.")
        return
    semanas = sorted(df["semana"].unique())
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True, constrained_layout=True)
    for i, semana in enumerate(semanas[:8]):  # máximo 8 semanas
        ax = axs[i // 4, i % 4]
        df_semana = df[(df["semana"] == semana) & (df["segundos"] <= 30)]
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
    if not {"pitch","roll"}.issubset(df.columns):
        st.warning("No hay datos de pitch/roll disponibles.")
        return
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(x="semana", y="pitch", data=df, ax=axs[0])
    axs[0].set_title("Pitch por semana")
    sns.boxplot(x="semana", y="roll", data=df, ax=axs[1])
    axs[1].set_title("Roll por semana")
    st.pyplot(fig)

# 10. Histograma de repeticiones
def histograma_repeticiones(df, ejercicio=None):
    st.subheader("📊 Histograma de repeticiones")
    if "repeticiones" not in df.columns or df.empty:
        st.warning("No hay datos de repeticiones disponibles.")
        return
    repes = pd.to_numeric(df["repeticiones"], errors="coerce").dropna()
    if repes.empty:
        st.warning("No hay datos para este ejercicio")
        return
    bins = range(int(repes.min()), int(repes.max()) + 2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(repes, bins=bins, color="#F2AB6D", rwidth=0.85)
    ax.set_title(f"Histograma de repeticiones {f'({ejercicio})' if ejercicio and ejercicio!='Todos' else ''}")
    ax.set_xlabel("Repeticiones")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

# 11. Histograma de peso 
def histograma_peso(df, ejercicio=None):
    st.subheader(f"🏋️ Histograma de carga {f'({ejercicio})' if ejercicio and ejercicio!='Todos' else ''}")
    pesos = pd.to_numeric(df["peso"], errors="coerce").dropna()
    if pesos.empty:
        st.warning("No hay datos para este ejercicio")
        return
    bins = range(int(pesos.min()), int(pesos.max()) + 2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pesos, bins=bins, color="#F2AB6D", rwidth=0.85)  # ← aquí estaba el error
    ax.set_title("Histograma de carga")
    ax.set_xlabel("Peso (kg)")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

