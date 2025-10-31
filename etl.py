import pandas as pd
from pathlib import Path

def rellenar_y_agregar(archivo_csv, dataset_csv, peso, serie, repeticiones, ejercicio, id_ejercicio, semana, rutina):
    """
    Procesa un archivo CSV del reloj, completa los datos de entrenamiento y lo añade al dataset consolidado.
    """

    # Leer CSV original
    df = pd.read_csv(archivo_csv)

    # Eliminar columna 'time' si existe
    if 'time' in df.columns:
        df.drop(columns=['time'], inplace=True)

    # Filtrar segundos_elapsed si existe
    if 'seconds_elapsed' in df.columns:
        duracion_total = df["seconds_elapsed"].max()
        df = df[(df["seconds_elapsed"] > 2) & (df["seconds_elapsed"] < duracion_total - 2)]

    # Añadir columnas de entrenamiento
    df["peso"] = peso
    df["serie"] = serie
    df["repeticiones"] = repeticiones
    df["ejercicio"] = ejercicio
    df["id_ejercicio"] = id_ejercicio
    df["semana"] = semana
    df["rutina"] = rutina

    # Añadir columnas faltantes si el archivo original tenía más
    encabezados_originales = pd.read_csv(archivo_csv).columns.tolist()
    for encabezado in encabezados_originales:
        if encabezado not in df.columns:
            df[encabezado] = None

    # Guardar en dataset consolidado
    if not Path(dataset_csv).exists():
        df.to_csv(dataset_csv, index=False)
    else:
        df.to_csv(dataset_csv, mode='a', header=False, index=False)

    return df  # Devuelve el dataframe procesado para mostrar en Streamlit
