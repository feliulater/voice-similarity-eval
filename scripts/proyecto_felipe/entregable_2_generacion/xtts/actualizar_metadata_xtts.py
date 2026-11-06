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
metadata_xtts_csv = project_dir / "metadata" / "metadata_voces_sinteticas_xtts.csv"

xtts_root = project_dir / "voces_sinteticas" / "xtts"
xtts_raw_root = project_dir / "voces_sinteticas" / "xtts_raw"

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
        if row["sistema_tts"] == "xtts":
            output_path = project_dir / row["archivo_salida"]

            if output_path.exists() and output_path.stat().st_size > 0:
                row["estado"] = "generado_postprocesado"
                row["observaciones"] = "Audio generado con XTTS y limpiado mediante postprocesamiento automático."
            else:
                row["estado"] = "pendiente"
                row["observaciones"] = "Archivo XTTS no encontrado."

        rows.append(row)

with open(plan_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Crear metadata específica de XTTS
xtts_rows = []

for wav_path in sorted(xtts_root.glob("spk*/**/*.wav")):
    speaker_id = wav_path.parent.name
    filename = wav_path.name

    # Ejemplo: spk001_xtts_prompt001.wav
    parts = filename.replace(".wav", "").split("_")
    prompt_id = parts[-1]

    raw_path = xtts_raw_root / speaker_id / filename

    xtts_rows.append({
        "synthetic_id": f"syn_{speaker_id}_xtts_{prompt_id}",
        "speaker_id": speaker_id,
        "sistema_tts": "xtts",
        "tipo_audio": "sintetico",
        "prompt_id": prompt_id,
        "archivo_raw": str(raw_path.relative_to(project_dir)).replace("\\", "/") if raw_path.exists() else "",
        "archivo_limpio": str(wav_path.relative_to(project_dir)).replace("\\", "/"),
        "sample_rate": 16000,
        "canales": "mono",
        "formato": "wav",
        "duracion_segundos": get_duration(wav_path),
        "estado": "generado_postprocesado",
        "observaciones": "Audio XTTS postprocesado para eliminar artefactos finales aislados."
    })

with open(metadata_xtts_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "synthetic_id",
        "speaker_id",
        "sistema_tts",
        "tipo_audio",
        "prompt_id",
        "archivo_raw",
        "archivo_limpio",
        "sample_rate",
        "canales",
        "formato",
        "duracion_segundos",
        "estado",
        "observaciones"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(xtts_rows)

print(f"Plan actualizado: {plan_csv}")
print(f"Metadata XTTS creada: {metadata_xtts_csv}")
print(f"Audios XTTS registrados: {len(xtts_rows)}")
