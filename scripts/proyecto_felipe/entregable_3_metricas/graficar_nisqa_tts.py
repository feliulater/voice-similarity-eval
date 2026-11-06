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
    "nisqa_tts_por_audio.csv"
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
# BOXPLOT
# ============================================================

datos = [
    df.loc[
        df["sistema_tts"] == sistema,
        "mos_pred"
    ].values
    for sistema in orden
]

plt.figure(figsize=(9, 6))

plt.boxplot(
    datos,
    tick_labels=[etiquetas[s] for s in orden],
    showmeans=True
)

plt.ylabel("MOS estimado de naturalidad")
plt.xlabel("Sistema TTS")
plt.title("Distribución de naturalidad estimada con NISQA-TTS")
plt.ylim(1, 5)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

ruta_boxplot = os.path.join(
    GRAFICAS,
    "boxplot_nisqa_tts_por_sistema.png"
)

plt.savefig(
    ruta_boxplot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# RESUMEN
# ============================================================

resumen = (
    df.groupby("sistema_tts")["mos_pred"]
    .agg(["count", "mean", "std", "median", "min", "max"])
    .reindex(orden)
)

resumen.to_csv(
    os.path.join(
        TABLAS,
        "resumen_nisqa_tts_graficas.csv"
    )
)

# ============================================================
# BARRAS
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    [etiquetas[s] for s in orden],
    resumen["mean"],
    yerr=resumen["std"],
    capsize=6
)

plt.ylabel("MOS estimado promedio")
plt.xlabel("Sistema TTS")
plt.title(
    "Naturalidad estimada por sistema TTS\n"
    "NISQA-TTS, barras de error: ±1 desviación estándar"
)

plt.ylim(1, 5)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

ruta_barras = os.path.join(
    GRAFICAS,
    "barras_promedio_nisqa_tts_por_sistema.png"
)

plt.savefig(
    ruta_barras,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Graficas generadas:")
print(ruta_boxplot)
print(ruta_barras)

print()
print("Resumen:")
print(
    resumen.rename(index=etiquetas).to_string()
)
print()
