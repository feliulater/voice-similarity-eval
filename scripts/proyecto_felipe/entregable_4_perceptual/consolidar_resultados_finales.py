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

E3 = ROOT / "entregable_3_metricas"
E4 = ROOT / "entregable_4_analisis_final"

PER = E4 / "evaluacion_perceptual"

OBJETIVAS = (
    E3
    / "tablas_resumen"
    / "resumen_general_metricas_entregable3.csv"
)

PERCEPTUALES = (
    PER
    / "tablas_resumen"
    / "resumen_perceptual_global.csv"
)

PERCEPTUALES_IC = (
    PER
    / "tablas_resumen"
    / "resumen_perceptual_participantes_ic95.csv"
)

SALIDA = PER / "tablas_resumen"

SALIDA.mkdir(
    parents=True,
    exist_ok=True
)

ORDEN = [
    "xtts",
    "f5tts",
    "openvoice",
    "elevenlabs",
]

NOMBRES = {
    "xtts": "XTTS",
    "f5tts": "F5-TTS",
    "openvoice": "OpenVoice",
    "elevenlabs": "ElevenLabs",
}


# ============================================================
# FUNCIONES
# ============================================================

def normalizar_sistema(nombre):

    texto = str(nombre).strip().lower()

    texto = (
        texto
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    mapa = {
        "xtts": "xtts",
        "f5tts": "f5tts",
        "openvoice": "openvoice",
        "elevenlabs": "elevenlabs",
    }

    return mapa.get(texto, texto)


# ============================================================
# LEER ARCHIVOS
# ============================================================

objetivas = pd.read_csv(
    OBJETIVAS,
    encoding="utf-8-sig"
)

perceptuales = pd.read_csv(
    PERCEPTUALES,
    encoding="utf-8-sig"
)

perceptuales_ic = pd.read_csv(
    PERCEPTUALES_IC,
    encoding="utf-8-sig"
)


# ============================================================
# NORMALIZAR NOMBRES DE SISTEMA
# ============================================================

for df in [
    objetivas,
    perceptuales,
    perceptuales_ic
]:

    df["sistema_tts"] = (
        df["sistema_tts"]
        .apply(normalizar_sistema)
    )


# ============================================================
# VALIDACIONES
# ============================================================

columnas_objetivas = [
    "sistema_tts",
    "ecapa_sim_media",
    "ecapa_sim_std",
    "eer_porcentaje",
    "auc_roc",
    "mfcc_dtw_media",
    "mfcc_dtw_std",
    "nisqa_mos_media",
    "nisqa_mos_std",
]

columnas_perceptuales = [
    "sistema_tts",
    "similitud_media",
    "similitud_std",
    "naturalidad_media",
    "naturalidad_std",
]

columnas_ic = [
    "sistema_tts",
    "n_participantes",
    "similitud_ci95",
    "naturalidad_ci95",
]

for columna in columnas_objetivas:

    if columna not in objetivas.columns:
        raise ValueError(
            f"Falta columna objetiva: {columna}"
        )

for columna in columnas_perceptuales:

    if columna not in perceptuales.columns:
        raise ValueError(
            f"Falta columna perceptual: {columna}"
        )

for columna in columnas_ic:

    if columna not in perceptuales_ic.columns:
        raise ValueError(
            f"Falta columna IC95: {columna}"
        )


# ============================================================
# SELECCIONAR COLUMNAS
# ============================================================

obj = objetivas[
    columnas_objetivas
].copy()

per = perceptuales[
    columnas_perceptuales
].copy()

ic = perceptuales_ic[
    columnas_ic
].copy()


# ============================================================
# COMBINAR
# ============================================================

integrado = pd.merge(
    obj,
    per,
    on="sistema_tts",
    how="inner"
)

integrado = pd.merge(
    integrado,
    ic,
    on="sistema_tts",
    how="inner"
)

integrado = (
    integrado
    .set_index("sistema_tts")
    .reindex(ORDEN)
    .reset_index()
)

if len(integrado) != 4:

    raise ValueError(
        f"Se esperaban 4 sistemas y se obtuvieron "
        f"{len(integrado)}."
    )


# ============================================================
# REDONDEAR VERSION NUMERICA
# ============================================================

numerico = integrado.copy()

columnas_redondeo = [
    "ecapa_sim_media",
    "ecapa_sim_std",
    "eer_porcentaje",
    "auc_roc",
    "mfcc_dtw_media",
    "mfcc_dtw_std",
    "nisqa_mos_media",
    "nisqa_mos_std",
    "similitud_media",
    "similitud_std",
    "similitud_ci95",
    "naturalidad_media",
    "naturalidad_std",
    "naturalidad_ci95",
]

numerico[columnas_redondeo] = (
    numerico[columnas_redondeo]
    .round(4)
)

numerico.insert(
    1,
    "sistema",
    numerico["sistema_tts"].map(NOMBRES)
)


# ============================================================
# TABLA FORMATEADA PARA INFORME
# ============================================================

filas = []

for _, r in integrado.iterrows():

    filas.append({

        "Sistema":
            NOMBRES[r["sistema_tts"]],

        "ECAPA similitud":
            f'{r["ecapa_sim_media"]:.3f} ± '
            f'{r["ecapa_sim_std"]:.3f}',

        "EER (%)":
            f'{r["eer_porcentaje"]:.2f}',

        "AUC ROC":
            f'{r["auc_roc"]:.3f}',

        "MFCC-DTW":
            f'{r["mfcc_dtw_media"]:.2f} ± '
            f'{r["mfcc_dtw_std"]:.2f}',

        "NISQA-TTS":
            f'{r["nisqa_mos_media"]:.2f} ± '
            f'{r["nisqa_mos_std"]:.2f}',

        "Similitud perceptual":
            f'{r["similitud_media"]:.2f} ± '
            f'{r["similitud_std"]:.2f}',

        "IC95 similitud":
            f'±{r["similitud_ci95"]:.2f}',

        "Naturalidad perceptual":
            f'{r["naturalidad_media"]:.2f} ± '
            f'{r["naturalidad_std"]:.2f}',

        "IC95 naturalidad":
            f'±{r["naturalidad_ci95"]:.2f}',

        "Participantes":
            int(r["n_participantes"]),
    })

tabla_formateada = pd.DataFrame(filas)


# ============================================================
# RANGOS DESCRIPTIVOS
#
# Solo sirven para visualizar concordancias y discrepancias.
# No constituyen una puntuacion global.
# ============================================================

rangos = integrado[
    [
        "sistema_tts",
        "ecapa_sim_media",
        "eer_porcentaje",
        "mfcc_dtw_media",
        "nisqa_mos_media",
        "similitud_media",
        "naturalidad_media",
    ]
].copy()

rangos["sistema"] = (
    rangos["sistema_tts"]
    .map(NOMBRES)
)

# Mayor = mejor
rangos["rango_ecapa"] = (
    rangos["ecapa_sim_media"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

# Menor = mejor
rangos["rango_eer"] = (
    rangos["eer_porcentaje"]
    .rank(
        ascending=True,
        method="min"
    )
    .astype(int)
)

# Menor = mejor
rangos["rango_mfcc_dtw"] = (
    rangos["mfcc_dtw_media"]
    .rank(
        ascending=True,
        method="min"
    )
    .astype(int)
)

# Mayor = mejor
rangos["rango_nisqa"] = (
    rangos["nisqa_mos_media"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

# Mayor = mejor
rangos["rango_similitud_perceptual"] = (
    rangos["similitud_media"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

# Mayor = mejor
rangos["rango_naturalidad_perceptual"] = (
    rangos["naturalidad_media"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

rangos = rangos[
    [
        "sistema",
        "rango_ecapa",
        "rango_eer",
        "rango_mfcc_dtw",
        "rango_nisqa",
        "rango_similitud_perceptual",
        "rango_naturalidad_perceptual",
    ]
]


# ============================================================
# GUARDAR
# ============================================================

archivo_numerico = (
    SALIDA
    / "resumen_integrado_objetivo_perceptual.csv"
)

archivo_formateado = (
    SALIDA
    / "tabla_integrada_objetivo_perceptual.csv"
)

archivo_rangos = (
    SALIDA
    / "rangos_comparativos_metricas.csv"
)

numerico.to_csv(
    archivo_numerico,
    index=False,
    encoding="utf-8-sig"
)

tabla_formateada.to_csv(
    archivo_formateado,
    index=False,
    encoding="utf-8-sig"
)

rangos.to_csv(
    archivo_rangos,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

print()
print("============================================================")
print(" RESUMEN INTEGRADO OBJETIVO + PERCEPTUAL")
print("============================================================")
print()

print(
    tabla_formateada.to_string(
        index=False
    )
)

print()
print("============================================================")
print(" RANGOS DESCRIPTIVOS")
print("============================================================")
print()

print(
    rangos.to_string(
        index=False
    )
)

print()
print("============================================================")
print(" ARCHIVOS GENERADOS")
print("============================================================")
print()

print(archivo_numerico)
print(archivo_formateado)
print(archivo_rangos)

print()
print(
    "IMPORTANTE: los rangos son descriptivos y no deben "
    "combinarse en una puntuacion global unica."
)

print()
print("Consolidacion completada correctamente.")
print()
