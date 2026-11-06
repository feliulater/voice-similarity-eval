import os
import pandas as pd
import matplotlib.pyplot as plt


ROOT = os.environ.get(
    "PROYECTO_TTS_ROOT",
    os.path.expanduser("~/Proyecto_Electrico_Voces_TTS"),
)

CSV = os.path.join(
    ROOT,
    "entregable_3_metricas",
    "resultados_csv",
    "similitud_coseno_por_audio.csv"
)

GRAFICAS = os.path.join(
    ROOT,
    "entregable_3_metricas",
    "graficas"
)

TABLAS = os.path.join(
    ROOT,
    "entregable_3_metricas",
    "tablas_resumen"
)

os.makedirs(GRAFICAS, exist_ok=True)
os.makedirs(TABLAS, exist_ok=True)

df = pd.read_csv(CSV)

df = df[df["estado"] == "ok"].copy()

orden = [
    "xtts",
    "f5tts",
    "openvoice",
    "elevenlabs"
]

etiquetas = {
    "xtts": "XTTS",
    "f5tts": "F5-TTS",
    "openvoice": "OpenVoice",
    "elevenlabs": "ElevenLabs"
}


# ============================================================
# 1. BOXPLOT
# ============================================================

datos = [
    df.loc[
        df["sistema_tts"] == sistema,
        "similitud_coseno"
    ].values
    for sistema in orden
]

plt.figure(figsize=(9, 6))

plt.boxplot(
    datos,
    tick_labels=[etiquetas[s] for s in orden],
    showmeans=True
)

plt.ylabel("Similitud coseno")
plt.xlabel("Sistema TTS")
plt.title("Distribución de similitud de hablante por sistema TTS")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

ruta_boxplot = os.path.join(
    GRAFICAS,
    "boxplot_similitud_coseno_por_sistema.png"
)

plt.savefig(
    ruta_boxplot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. PROMEDIO + DESVIACION ESTANDAR
# ============================================================

resumen = (
    df.groupby("sistema_tts")
    ["similitud_coseno"]
    .agg(["count", "mean", "std", "median", "min", "max"])
    .reindex(orden)
)

resumen.to_csv(
    os.path.join(
        TABLAS,
        "resumen_similitud_ecapa_graficas.csv"
    )
)

plt.figure(figsize=(9, 6))

plt.bar(
    [etiquetas[s] for s in orden],
    resumen["mean"],
    yerr=resumen["std"],
    capsize=6
)

plt.ylabel("Similitud coseno promedio")
plt.xlabel("Sistema TTS")
plt.title("Similitud promedio de hablante por sistema TTS")
plt.ylim(0, 0.8)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

ruta_barras = os.path.join(
    GRAFICAS,
    "barras_promedio_similitud_por_sistema.png"
)

plt.savefig(
    ruta_barras,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("Graficas generadas correctamente:")
print()
print(ruta_boxplot)
print(ruta_barras)

print()
print("Resumen utilizado:")
print(
    resumen.rename(
        index=etiquetas
    ).to_string()
)
print()
