import pandas as pd
import numpy as np

def transformar_dataset(ruta_csv="data/registro.csv", salida_csv="data/registro_def.csv"):
    """
    Aplica limpieza, transformación y enriquecimiento al dataset de entrenamiento.
    """

    # Leer archivo
    df = pd.read_csv(ruta_csv)

    # Eliminar columnas innecesarias
    columnas_a_eliminar = [
        "time", "rotationRateX", "rotationRateY", "rotationRateZ",
        "gravityX", "gravityY", "gravityZ",
        "quaternionW", "quaternionX", "quaternionY", "quaternionZ"
    ]
    df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns], inplace=True)

    # Renombrar columnas si existen
    df.rename(columns={
        "seconds_elapsed": "segundos",
        "accelerationX": "accX",
        "accelerationY": "accY",
        "accelerationZ": "accZ"
    }, inplace=True)

    # Eliminar ejercicios específicos (ID 10)
    if "id_ejercicio" in df.columns:
        df = df[df["id_ejercicio"] != 10]

    # Convertir radianes a grados
    def radianes_a_grados(radianes):
        return radianes * (180 / 3.14)

    for col in ["pitch", "roll", "yaw"]:
        if col in df.columns:
            df[f"{col}_grados"] = df[col].apply(radianes_a_grados)

    # Calcular duración media por ejercicio
    if "ejercicio" in df.columns and "segundos" in df.columns:
        media = df.groupby("ejercicio")["segundos"].mean()
        df["duracion_media"] = df["ejercicio"].map(media)

    # Calcular volumen total
    if all(col in df.columns for col in ["peso", "repeticiones", "serie"]):
        df["volumen_total"] = df["peso"] * df["repeticiones"] * df["serie"]

    # Guardar resultado
    df.to_csv(salida_csv, index=False)

    return df
