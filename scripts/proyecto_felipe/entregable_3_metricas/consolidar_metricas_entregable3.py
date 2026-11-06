import os
import pandas as pd

ROOT = os.environ.get(
    "PROYECTO_TTS_ROOT",
    os.path.expanduser("~/Proyecto_Electrico_Voces_TTS"),
)
E3 = os.path.join(ROOT, "entregable_3_metricas")

# ------------------------------------------------------------
# Archivos fuente
# ------------------------------------------------------------

sim_path = os.path.join(
    E3,
    "tablas_resumen",
    "resumen_similitud_por_sistema.csv"
)

eer_path = os.path.join(
    E3,
    "resultados_csv",
    "eer_por_sistema.csv"
)

mfcc_path = os.path.join(
    E3,
    "tablas_resumen",
    "resumen_distancia_cepstral_por_sistema.csv"
)

nisqa_path = os.path.join(
    E3,
    "tablas_resumen",
    "resumen_nisqa_tts_por_sistema.csv"
)

qc_path = os.path.join(
    E3,
    "tablas_resumen",
    "resumen_control_calidad_audios.csv"
)

salida = os.path.join(
    E3,
    "tablas_resumen",
    "resumen_general_metricas_entregable3.csv"
)

# ------------------------------------------------------------
# Leer
# ------------------------------------------------------------

sim = pd.read_csv(sim_path)
eer = pd.read_csv(eer_path)
mfcc = pd.read_csv(mfcc_path)
nisqa = pd.read_csv(nisqa_path)
qc = pd.read_csv(qc_path)

# Solo fuentes sintéticas
qc = qc[
    qc["fuente"].isin(
        ["xtts", "f5tts", "openvoice", "elevenlabs"]
    )
].copy()

# ------------------------------------------------------------
# Seleccionar y renombrar
# ------------------------------------------------------------

sim = sim[
    [
        "sistema_tts",
        "promedio",
        "desviacion_estandar",
        "mediana"
    ]
].rename(
    columns={
        "promedio": "ecapa_sim_media",
        "desviacion_estandar": "ecapa_sim_std",
        "mediana": "ecapa_sim_mediana"
    }
)

eer = eer[
    [
        "sistema_tts",
        "eer_porcentaje",
        "auc",
        "threshold_eer"
    ]
].rename(
    columns={
        "eer_porcentaje": "eer_porcentaje",
        "auc": "auc_roc",
        "threshold_eer": "umbral_eer"
    }
)

mfcc = mfcc[
    [
        "sistema_tts",
        "promedio",
        "desviacion_estandar",
        "mediana"
    ]
].rename(
    columns={
        "promedio": "mfcc_dtw_media",
        "desviacion_estandar": "mfcc_dtw_std",
        "mediana": "mfcc_dtw_mediana"
    }
)

nisqa = nisqa[
    [
        "sistema_tts",
        "promedio",
        "desviacion_estandar",
        "mediana"
    ]
].rename(
    columns={
        "promedio": "nisqa_mos_media",
        "desviacion_estandar": "nisqa_mos_std",
        "mediana": "nisqa_mos_mediana"
    }
)

qc = qc[
    [
        "fuente",
        "duracion_media_s",
        "duracion_trim_media_s",
        "silencio_extremos_medio_s",
        "pico_medio",
        "rms_medio",
        "clipping_maximo"
    ]
].rename(
    columns={
        "fuente": "sistema_tts"
    }
)

# ------------------------------------------------------------
# Fusionar
# ------------------------------------------------------------

general = (
    sim
    .merge(eer, on="sistema_tts", how="outer")
    .merge(mfcc, on="sistema_tts", how="outer")
    .merge(nisqa, on="sistema_tts", how="outer")
    .merge(qc, on="sistema_tts", how="outer")
)

orden = [
    "xtts",
    "f5tts",
    "openvoice",
    "elevenlabs"
]

general["orden"] = general["sistema_tts"].apply(
    lambda x: orden.index(x)
)

general = (
    general
    .sort_values("orden")
    .drop(columns=["orden"])
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# Guardar
# ------------------------------------------------------------

general.to_csv(
    salida,
    index=False
)

print()
print("============================================================")
print(" RESUMEN GENERAL - ENTREGABLE 3")
print("============================================================")
print()

print(
    general.to_string(
        index=False
    )
)

print()
print("Guardado en:")
print(salida)

print()
print("Interpretacion de métricas:")
print("  ECAPA similitud : mayor = mayor similitud de hablante")
print("  EER              : menor = mejor separación de identidad")
print("  AUC ROC          : mayor = mejor separación de identidad")
print("  MFCC-DTW         : menor = mayor semejanza espectral")
print("  NISQA-TTS MOS    : mayor = mayor naturalidad estimada")
print()
