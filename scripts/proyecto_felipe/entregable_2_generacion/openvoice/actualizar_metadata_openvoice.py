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
metadata_openvoice_csv = project_dir / "metadata" / "metadata_voces_sinteticas_openvoice.csv"

openvoice_root = project_dir / "voces_sinteticas" / "openvoice"
base_root = project_dir / "voces_sinteticas" / "openvoice_intermedios"

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
        if row["sistema_tts"] == "openvoice":
            output_path = project_dir / row["archivo_salida"]

            if output_path.exists() and output_path.stat().st_size > 0:
                row["estado"] = "generado"
                row["observaciones"] = "Audio generado con OpenVoice V2 usando voz base MeloTTS ES; conserva rasgos de acento/prosodia de la voz base."
            else:
                row["estado"] = "pendiente"
                row["observaciones"] = "Archivo OpenVoice no encontrado."

        rows.append(row)

with open(plan_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Crear metadata específica de OpenVoice
openvoice_rows = []

for wav_path in sorted(openvoice_root.glob("spk*/**/*_openvoice_prompt*.wav")):
    speaker_id = wav_path.parent.name
    filename = wav_path.name

    # Ejemplo: spk001_openvoice_prompt001.wav
    prompt_id = filename.replace(".wav", "").split("_")[-1]

    base_audio = base_root / speaker_id / f"{speaker_id}_openvoice_base_{prompt_id}.wav"

    openvoice_rows.append({
        "synthetic_id": f"syn_{speaker_id}_openvoice_{prompt_id}",
        "speaker_id": speaker_id,
        "sistema_tts": "openvoice",
        "tipo_audio": "sintetico",
        "prompt_id": prompt_id,
        "archivo_salida": str(wav_path.relative_to(project_dir)).replace("\\", "/"),
        "archivo_base_intermedio": str(base_audio.relative_to(project_dir)).replace("\\", "/") if base_audio.exists() else "",
        "sample_rate": 16000,
        "canales": "mono",
        "formato": "wav",
        "duracion_segundos": get_duration(wav_path),
        "estado": "generado",
        "postprocesamiento": "no_aplicado",
        "observaciones": "OpenVoice V2 con voz base MeloTTS ES; audio inteligible, pero con acento/prosodia influenciada por la voz base."
    })

with open(metadata_openvoice_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "synthetic_id",
        "speaker_id",
        "sistema_tts",
        "tipo_audio",
        "prompt_id",
        "archivo_salida",
        "archivo_base_intermedio",
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
    writer.writerows(openvoice_rows)

print(f"Plan actualizado: {plan_csv}")
print(f"Metadata OpenVoice creada: {metadata_openvoice_csv}")
print(f"Audios OpenVoice registrados: {len(openvoice_rows)}")
