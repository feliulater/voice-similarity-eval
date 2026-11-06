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
    "pares_verificacion_ecapa.csv"
)

GRAFICAS = os.path.join(
    ROOT,
    "entregable_3_metricas",
    "graficas"
)

os.makedirs(GRAFICAS, exist_ok=True)

df = pd.read_csv(CSV)

sistemas = [
    ("xtts", "XTTS"),
    ("f5tts", "F5-TTS"),
    ("openvoice", "OpenVoice"),
    ("elevenlabs", "ElevenLabs"),
]

# Los mismos límites y bins en todas las figuras
bins = 35
rango = (-0.2, 0.8)

for sistema, etiqueta in sistemas:

    datos = df[df["sistema_tts"] == sistema]

    positivos = datos[
        datos["label"] == 1
    ]["similitud_coseno"]

    negativos = datos[
        datos["label"] == 0
    ]["similitud_coseno"]

    plt.figure(figsize=(9, 6))

    plt.hist(
        negativos,
        bins=bins,
        range=rango,
        alpha=0.6,
        density=True,
        label="Pares negativos"
    )

    plt.hist(
        positivos,
        bins=bins,
        range=rango,
        alpha=0.6,
        density=True,
        label="Pares positivos"
    )

    plt.axvline(
        positivos.mean(),
        linestyle="--",
        linewidth=1.5,
        label=f"Media positiva = {positivos.mean():.3f}"
    )

    plt.axvline(
        negativos.mean(),
        linestyle="--",
        linewidth=1.5,
        label=f"Media negativa = {negativos.mean():.3f}"
    )

    plt.xlabel("Similitud coseno")
    plt.ylabel("Densidad")
    plt.title(
        f"Distribución de scores de verificación - {etiqueta}"
    )

    plt.xlim(rango)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    salida = os.path.join(
        GRAFICAS,
        f"distribucion_scores_{sistema}.png"
    )

    plt.savefig(
        salida,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Generada:", salida)

print()
print("Proceso finalizado.")
