import os
import pandas as pd

ROOT = os.environ.get(
    "PROYECTO_TTS_ROOT",
    os.path.expanduser("~/Proyecto_Electrico_Voces_TTS"),
)

MANIFEST = os.path.join(
    ROOT,
    "entregable_3_metricas",
    "config",
    "manifest_evaluacion.csv"
)

SALIDA = os.path.join(
    ROOT,
    "entregable_3_metricas",
    "config",
    "nisqa_tts_input.csv"
)

df = pd.read_csv(MANIFEST)

sistemas = {
    "xtts": "xtts_path",
    "f5tts": "f5tts_path",
    "openvoice": "openvoice_path",
    "elevenlabs": "elevenlabs_path",
}

filas = []

for _, row in df.iterrows():

    for sistema, columna in sistemas.items():

        ruta_relativa = row[columna]

        ruta_absoluta = os.path.join(
            ROOT,
            ruta_relativa
        )

        filas.append({
            "sample_id": row["sample_id"],
            "speaker_id": row["speaker_id"],
            "prompt_id": row["prompt_id"],
            "sistema_tts": sistema,
            "filepath": ruta_absoluta,
        })

out = pd.DataFrame(filas)

out.to_csv(
    SALIDA,
    index=False
)

print()
print("========================================")
print(" CSV NISQA-TTS PREPARADO")
print("========================================")
print()

print("Filas totales:", len(out))

print()
print("Filas por sistema:")
print(
    out["sistema_tts"]
    .value_counts()
    .sort_index()
)

print()
print(
    "Archivos existentes:",
    out["filepath"].apply(os.path.exists).sum(),
    "/",
    len(out)
)

print()
print("Guardado en:")
print(SALIDA)
print()
