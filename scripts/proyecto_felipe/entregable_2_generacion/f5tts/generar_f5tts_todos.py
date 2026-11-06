from pathlib import Path
import csv
import time
import subprocess
import os

project_dir = Path(
    os.environ.get(
        "PROYECTO_TTS_E2_DIR",
        Path.home() / "Proyecto_Electrico_Voces_TTS" / "entregable_2_voces",
    )
)

prompts_csv = project_dir / "metadata" / "prompts_final_entregable2.csv"
f5_refs_csv = project_dir / "metadata" / "f5tts_referencias.csv"

output_root = project_dir / "voces_sinteticas" / "f5tts"
metadata_dir = project_dir / "metadata"
log_csv = metadata_dir / "log_generacion_f5tts.csv"

speakers = ["spk001", "spk002", "spk003", "spk004", "spk005"]

# Cachés configurables mediante variable de entorno
cache_root = os.environ.get(
    "TTS_MODEL_CACHE",
    str(Path.home() / ".cache" / "tts_model_cache"),
)
env = os.environ.copy()
env["HF_HOME"] = f"{cache_root}/huggingface"
env["TRANSFORMERS_CACHE"] = f"{cache_root}/huggingface"
env["HUGGINGFACE_HUB_CACHE"] = f"{cache_root}/huggingface"
env["TORCH_HOME"] = f"{cache_root}/torch"
env["XDG_CACHE_HOME"] = cache_root

# Leer prompts
prompts = []
with open(prompts_csv, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompts.append({
            "prompt_id": row["prompt_id"],
            "texto": row["texto"]
        })

# Leer textos de referencia F5
ref_texts = {}
ref_audios = {}

with open(f5_refs_csv, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        spk = row["speaker_id"]
        ref_texts[spk] = row["texto_referencia"]
        ref_audios[spk] = project_dir / row["archivo_referencia"]

print(f"Prompts cargados: {len(prompts)}")
print(f"Speakers: {len(speakers)}")

output_root.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)

for spk in speakers:
    (output_root / spk).mkdir(parents=True, exist_ok=True)

# Crear log
if not log_csv.exists():
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "speaker_id",
            "prompt_id",
            "texto",
            "archivo_salida",
            "referencia_usada",
            "texto_referencia",
            "estado",
            "duracion_generacion_seg",
            "error",
            "observaciones"
        ])

total = len(speakers) * len(prompts)
contador = 0

for spk in speakers:
    ref_audio = ref_audios[spk]
    ref_text = ref_texts[spk]

    for prompt in prompts:
        contador += 1

        prompt_id = prompt["prompt_id"]
        texto = prompt["texto"]

        output_dir = output_root / spk
        output_file = f"{spk}_f5tts_{prompt_id}.wav"
        output_path = output_dir / output_file

        print(f"\n[{contador}/{total}] {spk} - {prompt_id}")
        print("Texto:", texto)
        print("Salida:", output_path)

        if output_path.exists() and output_path.stat().st_size > 0:
            print("Ya existe, se omite.")
            estado = "omitido_existente"
            error = ""
            duracion = 0.0
        else:
            cmd = [
                "f5-tts_infer-cli",
                "--model", "F5TTS_v1_Base",
                "--ref_audio", str(ref_audio),
                "--ref_text", ref_text,
                "--gen_text", texto,
                "--output_dir", str(output_dir),
                "--output_file", output_file,
                "--device", "cuda"
            ]

            inicio = time.time()

            try:
                result = subprocess.run(
                    cmd,
                    env=env,
                    text=True,
                    capture_output=True
                )

                duracion = round(time.time() - inicio, 3)

                if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                    estado = "generado"
                    error = ""
                    print(f"Generado en {duracion} s")
                else:
                    estado = "error"
                    error = result.stderr[-1000:] if result.stderr else "Error desconocido"
                    print("ERROR:")
                    print(error)

            except Exception as e:
                duracion = round(time.time() - inicio, 3)
                estado = "error"
                error = repr(e)
                print("ERROR:", error)

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                spk,
                prompt_id,
                texto,
                str(output_path.relative_to(project_dir)).replace("\\", "/"),
                str(ref_audio.relative_to(project_dir)).replace("\\", "/"),
                ref_text,
                estado,
                duracion,
                error,
                "Salida directa de F5-TTS; se documenta posible acento/prosodia extranjera en español."
            ])

print("\nProceso F5-TTS finalizado.")
print("Log:", log_csv)
