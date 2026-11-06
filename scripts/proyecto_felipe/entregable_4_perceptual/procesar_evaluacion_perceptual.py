from pathlib import Path
import os
import pandas as pd

# ============================================================
# CONFIGURACION
# ============================================================

ROOT = Path(
    os.environ.get(
        "PROYECTO_TTS_ROOT",
        Path.home() / "Proyecto_Electrico_Voces_TTS",
    )
)

PER = ROOT / "entregable_4_analisis_final" / "evaluacion_perceptual"

CRUDOS = PER / "resultados" / "crudos"
PROCESADOS = PER / "resultados" / "procesados"
CONFIG = PER / "config"

CLAVE = CONFIG / "asignacion_perceptual_interna.csv"

VERSIONES = ["A", "B", "C", "D"]
LABELS = ["A", "B", "C", "D"]

PROCESADOS.mkdir(parents=True, exist_ok=True)

# ============================================================
# LEER CLAVE INTERNA
# ============================================================

clave = pd.read_csv(CLAVE, encoding="utf-8-sig")

columnas_clave = [
    "version",
    "speaker_id",
    "speaker_label",
    "target_prompt_id",
    "reference_prompt_id",
    "blind_label",
    "sistema_tts",
]

faltantes = [c for c in columnas_clave if c not in clave.columns]

if faltantes:
    raise ValueError(
        f"Faltan columnas en asignacion_perceptual_interna.csv: {faltantes}"
    )

print()
print("============================================================")
print(" PROCESAMIENTO DE EVALUACION PERCEPTUAL")
print("============================================================")
print()

print(f"Filas de la clave interna: {len(clave)}")

if len(clave) != 80:
    raise ValueError(
        f"La clave interna deberia tener 80 filas y tiene {len(clave)}."
    )


filas = []
control_versiones = []

for version in VERSIONES:

    archivo = CRUDOS / f"respuestas_version_{version}.csv"

    if not archivo.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo: {archivo}"
        )

    respuestas = pd.read_csv(
        archivo,
        encoding="utf-8-sig"
    )

    print()
    print("------------------------------------------------------------")
    print(f"VERSION {version}")
    print("------------------------------------------------------------")
    print(f"Participantes: {len(respuestas)}")
    print(f"Columnas: {len(respuestas.columns)}")

    if len(respuestas) != 5:
        raise ValueError(
            f"La version {version} deberia tener 5 respuestas "
            f"y tiene {len(respuestas)}."
        )

    if len(respuestas.columns) != 41:
        raise ValueError(
            f"La version {version} deberia tener 41 columnas "
            f"y tiene {len(respuestas.columns)}."
        )

    # Ignorar la primera columna (marca temporal)
    columnas_respuesta = list(respuestas.columns[1:])

    if len(columnas_respuesta) != 40:
        raise ValueError(
            f"La version {version} no contiene las 40 "
            f"columnas perceptuales esperadas."
        )

    # --------------------------------------------------------
    # Crear IDs anonimos
    # A01-A05, B01-B05, etc.
    # --------------------------------------------------------

    for indice_participante, (_, respuesta) in enumerate(
        respuestas.iterrows(),
        start=1
    ):

        participante_id = f"{version}{indice_participante:02d}"

        # ----------------------------------------------------
        # Cinco voces
        # Cada voz ocupa exactamente 8 columnas:
        # 4 similitud + 4 naturalidad
        # ----------------------------------------------------

        for voz_num in range(1, 6):

            inicio = (voz_num - 1) * 8

            cols_similitud = columnas_respuesta[
                inicio : inicio + 4
            ]

            cols_naturalidad = columnas_respuesta[
                inicio + 4 : inicio + 8
            ]

            speaker_label = f"Voz_{voz_num}"

            for posicion, blind_label in enumerate(LABELS):

                # --------------------------------------------
                # Buscar sistema real en la clave interna
                # --------------------------------------------

                coincidencia = clave[
                    (clave["version"] == version) &
                    (clave["speaker_label"] == speaker_label) &
                    (clave["blind_label"] == blind_label)
                ]

                if len(coincidencia) != 1:
                    raise ValueError(
                        f"No se encontro una unica asignacion para "
                        f"version={version}, "
                        f"speaker_label={speaker_label}, "
                        f"blind_label={blind_label}"
                    )

                info = coincidencia.iloc[0]

                similitud = pd.to_numeric(
                    respuesta[cols_similitud[posicion]],
                    errors="coerce"
                )

                naturalidad = pd.to_numeric(
                    respuesta[cols_naturalidad[posicion]],
                    errors="coerce"
                )

                filas.append({
                    "participante_id": participante_id,
                    "version": version,
                    "speaker_id": info["speaker_id"],
                    "speaker_label": speaker_label,
                    "target_prompt_id": info["target_prompt_id"],
                    "reference_prompt_id": info["reference_prompt_id"],
                    "blind_label": blind_label,
                    "sistema_tts": info["sistema_tts"],
                    "similitud": similitud,
                    "naturalidad": naturalidad,
                })

    control_versiones.append({
        "version": version,
        "participantes": len(respuestas),
        "columnas_csv": len(respuestas.columns),
    })

# ============================================================
# CREAR DATAFRAME CONSOLIDADO
# ============================================================

evaluaciones = pd.DataFrame(filas)

print()
print("============================================================")
print(" VALIDACION DE BASE CONSOLIDADA")
print("============================================================")
print()

print(f"Filas totales: {len(evaluaciones)}")

if len(evaluaciones) != 400:
    raise ValueError(
        f"Se esperaban 400 evaluaciones y se obtuvieron "
        f"{len(evaluaciones)}."
    )

# ============================================================
# VALIDAR CALIFICACIONES
# ============================================================

faltantes_sim = evaluaciones["similitud"].isna().sum()
faltantes_nat = evaluaciones["naturalidad"].isna().sum()

print(f"Similitud faltante: {faltantes_sim}")
print(f"Naturalidad faltante: {faltantes_nat}")

if faltantes_sim > 0 or faltantes_nat > 0:
    raise ValueError(
        "Hay calificaciones faltantes o no numericas."
    )

fuera_sim = (
    ~evaluaciones["similitud"].between(1, 5)
).sum()

fuera_nat = (
    ~evaluaciones["naturalidad"].between(1, 5)
).sum()

print(f"Similitud fuera de 1-5: {fuera_sim}")
print(f"Naturalidad fuera de 1-5: {fuera_nat}")

if fuera_sim > 0 or fuera_nat > 0:
    raise ValueError(
        "Se encontraron calificaciones fuera del rango 1-5."
    )

# Convertir a enteros
evaluaciones["similitud"] = evaluaciones["similitud"].astype(int)
evaluaciones["naturalidad"] = evaluaciones["naturalidad"].astype(int)

# ============================================================
# CONTROLES DE BALANCE
# ============================================================

print()
print("Evaluaciones por version:")
print(
    evaluaciones["version"]
    .value_counts()
    .sort_index()
)

print()
print("Evaluaciones por sistema:")
print(
    evaluaciones["sistema_tts"]
    .value_counts()
    .sort_index()
)

print()
print("Evaluaciones por hablante:")
print(
    evaluaciones["speaker_id"]
    .value_counts()
    .sort_index()
)

print()
print("Sistema x etiqueta ciega:")
print(
    pd.crosstab(
        evaluaciones["sistema_tts"],
        evaluaciones["blind_label"]
    )
)

# ============================================================
# RESUMEN POR SISTEMA
# ============================================================

resumen_sistema = (
    evaluaciones
    .groupby("sistema_tts")
    .agg(
        n_evaluaciones=("similitud", "size"),

        similitud_media=("similitud", "mean"),
        similitud_std=("similitud", "std"),
        similitud_mediana=("similitud", "median"),
        similitud_min=("similitud", "min"),
        similitud_max=("similitud", "max"),

        naturalidad_media=("naturalidad", "mean"),
        naturalidad_std=("naturalidad", "std"),
        naturalidad_mediana=("naturalidad", "median"),
        naturalidad_min=("naturalidad", "min"),
        naturalidad_max=("naturalidad", "max"),
    )
    .reset_index()
)

columnas_numericas = resumen_sistema.select_dtypes(
    include="number"
).columns

resumen_sistema[columnas_numericas] = (
    resumen_sistema[columnas_numericas]
    .round(4)
)

# ============================================================
# RESUMEN POR SISTEMA Y HABLANTE
# ============================================================

resumen_sistema_hablante = (
    evaluaciones
    .groupby(
        ["sistema_tts", "speaker_id"]
    )
    .agg(
        n_evaluaciones=("similitud", "size"),

        similitud_media=("similitud", "mean"),
        similitud_std=("similitud", "std"),
        similitud_mediana=("similitud", "median"),

        naturalidad_media=("naturalidad", "mean"),
        naturalidad_std=("naturalidad", "std"),
        naturalidad_mediana=("naturalidad", "median"),
    )
    .reset_index()
)

columnas_numericas = resumen_sistema_hablante.select_dtypes(
    include="number"
).columns

resumen_sistema_hablante[columnas_numericas] = (
    resumen_sistema_hablante[columnas_numericas]
    .round(4)
)

# ============================================================
# DISTRIBUCION DE RESPUESTAS 1-5
# ============================================================

distribuciones = []

for sistema in sorted(evaluaciones["sistema_tts"].unique()):

    subset = evaluaciones[
        evaluaciones["sistema_tts"] == sistema
    ]

    for metrica in ["similitud", "naturalidad"]:

        conteos = (
            subset[metrica]
            .value_counts()
            .reindex(range(1, 6), fill_value=0)
        )

        total = conteos.sum()

        for puntuacion, cantidad in conteos.items():

            distribuciones.append({
                "sistema_tts": sistema,
                "metrica": metrica,
                "puntuacion": puntuacion,
                "cantidad": cantidad,
                "porcentaje": round(
                    100 * cantidad / total,
                    4
                ),
            })

distribucion_df = pd.DataFrame(distribuciones)

# ============================================================
# CONTROL DE PARTICIPANTES
# ============================================================

control_df = pd.DataFrame(control_versiones)

# ============================================================
# GUARDAR ARCHIVOS
# ============================================================

archivo_evaluaciones = (
    PROCESADOS /
    "evaluaciones_perceptuales_por_audio.csv"
)

archivo_resumen = (
    PROCESADOS /
    "resumen_perceptual_por_sistema.csv"
)

archivo_hablante = (
    PROCESADOS /
    "resumen_perceptual_por_sistema_hablante.csv"
)

archivo_distribucion = (
    PROCESADOS /
    "distribucion_calificaciones_perceptuales.csv"
)

archivo_control = (
    PROCESADOS /
    "control_respuestas_por_version.csv"
)

evaluaciones.to_csv(
    archivo_evaluaciones,
    index=False,
    encoding="utf-8-sig"
)

resumen_sistema.to_csv(
    archivo_resumen,
    index=False,
    encoding="utf-8-sig"
)

resumen_sistema_hablante.to_csv(
    archivo_hablante,
    index=False,
    encoding="utf-8-sig"
)

distribucion_df.to_csv(
    archivo_distribucion,
    index=False,
    encoding="utf-8-sig"
)

control_df.to_csv(
    archivo_control,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

print()
print("============================================================")
print(" RESUMEN PERCEPTUAL POR SISTEMA")
print("============================================================")
print()

print(
    resumen_sistema.to_string(
        index=False
    )
)

print()
print("============================================================")
print(" ARCHIVOS GENERADOS")
print("============================================================")
print()

print(archivo_evaluaciones)
print(archivo_resumen)
print(archivo_hablante)
print(archivo_distribucion)
print(archivo_control)

print()
print("Procesamiento completado correctamente.")
print()
