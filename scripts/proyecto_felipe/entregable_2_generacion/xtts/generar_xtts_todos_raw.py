from pathlib import Path
import os
import csv
import time
import traceback
import torch
from TTS.api import TTS

project_dir = Path(
    os.environ.get(
        "PROYECTO_TTS_E2_DIR",
        Path.home() / "Proyecto_Electrico_Voces_TTS" / "entregable_2_voces",
    )
)

prompts_csv = project_dir / "metadata" / "prompts_final_entregable2.csv"
output_root = project_dir / "voces_sinteticas" / "xtts_raw"
metadata_dir = project_dir / "metadata"
log_csv = metadata_dir / "log_generacion_xtts_raw.csv"

speakers = ["spk001", "spk002", "spk003", "spk004", "spk005"]

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Dispositivo:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Leer prompts
prompts = []
with open(prompts_csv, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompts.append({
            "prompt_id": row["prompt_id"],
            "texto": row["texto"]
        })

print(f"Prompts cargados: {len(prompts)}")

# Crear carpetas
output_root.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)

for spk in speakers:
    (output_root / spk).mkdir(parents=True, exist_ok=True)

# Inicializar XTTS una sola vez
print("Cargando XTTS v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
print("Modelo cargado.")

# Crear log si no existe
if not log_csv.exists():
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "speaker_id",
            "prompt_id",
            "texto",
            "archivo_salida",
            "referencia_usada",
            "estado",
            "duracion_generacion_seg",
            "error"
        ])

total = len(speakers) * len(prompts)
contador = 0

for spk in speakers:
    reference_wav = project_dir / "referencias_tts" / "xtts" / f"{spk}_reference_xtts.wav"

    for prompt in prompts:
        contador += 1

        prompt_id = prompt["prompt_id"]
        texto = prompt["texto"]

        output_wav = output_root / spk / f"{spk}_xtts_{prompt_id}.wav"

        print(f"\n[{contador}/{total}] {spk} - {prompt_id}")
        print("Texto:", texto)
        print("Salida:", output_wav)

        if output_wav.exists() and output_wav.stat().st_size > 0:
            print("Ya existe, se omite.")
            estado = "omitido_existente"
            error = ""
            duracion = 0.0
        else:
            inicio = time.time()
            try:
                tts.tts_to_file(
                    text=texto,
                    speaker_wav=str(reference_wav),
                    language="es",
                    file_path=str(output_wav),
                    split_sentences=False
                )
                duracion = round(time.time() - inicio, 3)
                estado = "generado"
                error = ""
                print(f"Generado en {duracion} s")
            except Exception as e:
                duracion = round(time.time() - inicio, 3)
                estado = "error"
                error = repr(e)
                print("ERROR:", error)
                traceback.print_exc()

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                spk,
                prompt_id,
                texto,
                str(output_wav.relative_to(project_dir)).replace("\\", "/"),
                str(reference_wav.relative_to(project_dir)).replace("\\", "/"),
                estado,
                duracion,
                error
            ])

print("\nProceso XTTS finalizado.")
print("Log:", log_csv)
