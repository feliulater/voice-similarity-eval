from pathlib import Path
import os
import csv
import subprocess

project_dir = Path(
    os.environ.get(
        "PROYECTO_TTS_E2_DIR",
        Path.home() / "Proyecto_Electrico_Voces_TTS" / "entregable_2_voces",
    )
)

plan_csv = project_dir / "metadata" / "plan_generacion_sintetica.csv"
metadata_f5_csv = project_dir / "metadata" / "metadata_voces_sinteticas_f5tts.csv"

f5_root = project_dir / "voces_sinteticas" / "f5tts"

def get_duration(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(path)
        ],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return ""
    return round(float(result.stdout.strip()), 3)

# Actualizar plan general
rows = []
with open(plan_csv, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    for row in reader:
        if row["sistema_tts"] == "f5tts":
            output_path = project_dir / row["archivo_salida"]

            if output_path.exists() and output_path.stat().st_size > 0:
                row["estado"] = "generado"
                row["observaciones"] = "Audio generado con F5-TTS sin postprocesamiento correctivo; se documenta posible acento/prosodia extranjera en español."
            else:
                row["estado"] = "pendiente"
                row["observaciones"] = "Archivo F5-TTS no encontrado."

        rows.append(row)

with open(plan_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Crear metadata específica de F5-TTS
f5_rows = []

for wav_path in sorted(f5_root.glob("spk*/**/*.wav")):
    speaker_id = wav_path.parent.name
    filename = wav_path.name

    # Ejemplo: spk001_f5tts_prompt001.wav
    parts = filename.replace(".wav", "").split("_")
    prompt_id = parts[-1]

    f5_rows.append({
        "synthetic_id": f"syn_{speaker_id}_f5tts_{prompt_id}",
        "speaker_id": speaker_id,
        "sistema_tts": "f5tts",
        "tipo_audio": "sintetico",
        "prompt_id": prompt_id,
        "archivo_salida": str(wav_path.relative_to(project_dir)).replace("\\", "/"),
        "sample_rate": 16000,
        "canales": "mono",
        "formato": "wav",
        "duracion_segundos": get_duration(wav_path),
        "estado": "generado",
        "postprocesamiento": "no_aplicado",
        "observaciones": "Salida directa de F5-TTS; audio inteligible en español, con posible acento/prosodia extranjera."
    })

with open(metadata_f5_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "synthetic_id",
        "speaker_id",
        "sistema_tts",
        "tipo_audio",
        "prompt_id",
        "archivo_salida",
        "sample_rate",
        "canales",
        "formato",
        "duracion_segundos",
        "estado",
        "postprocesamiento",
        "observaciones"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(f5_rows)

print(f"Plan actualizado: {plan_csv}")
print(f"Metadata F5-TTS creada: {metadata_f5_csv}")
print(f"Audios F5-TTS registrados: {len(f5_rows)}")
