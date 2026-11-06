from pathlib import Path
import os
import sys
import csv
import time
import torch
import unidic_lite

# =========================
# Rutas base
# =========================
project_dir = Path(
    os.environ.get(
        "PROYECTO_TTS_E2_DIR",
        Path.home() / "Proyecto_Electrico_Voces_TTS" / "entregable_2_voces",
    )
)
openvoice_dir = Path(
    os.environ.get(
        "OPENVOICE_DIR",
        Path.home() / "openvoice_workspace" / "OpenVoice",
    )
)
melotts_dir = Path(
    os.environ.get(
        "MELOTTS_DIR",
        Path.home() / "openvoice_workspace" / "MeloTTS",
    )
)

# Repos locales
sys.path.insert(0, str(openvoice_dir))
sys.path.insert(0, str(melotts_dir))

# Cachés configurables mediante variable de entorno
cache_root = os.environ.get(
    "TTS_MODEL_CACHE",
    str(Path.home() / ".cache" / "tts_model_cache"),
)
os.environ["HF_HOME"] = f"{cache_root}/huggingface"
os.environ["TRANSFORMERS_CACHE"] = f"{cache_root}/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{cache_root}/huggingface"
os.environ["TORCH_HOME"] = f"{cache_root}/torch"
os.environ["XDG_CACHE_HOME"] = cache_root

# MeCab / unidic-lite
os.environ["MECABRC"] = str(Path(unidic_lite.DICDIR) / "mecabrc")

from melo.api import TTS
from openvoice.api import ToneColorConverter
from openvoice import se_extractor

# =========================
# Configuración
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", device)

speakers = ["spk001", "spk002", "spk003", "spk004", "spk005"]

prompts_csv = project_dir / "metadata" / "prompts_final_entregable2.csv"

output_root = project_dir / "voces_sinteticas" / "openvoice"
base_root = project_dir / "voces_sinteticas" / "openvoice_intermedios"
metadata_dir = project_dir / "metadata"
log_csv = metadata_dir / "log_generacion_openvoice.csv"

output_root.mkdir(parents=True, exist_ok=True)
base_root.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)

# Checkpoints OpenVoice V2
ckpt_converter = openvoice_dir / "checkpoints_v2" / "converter"
source_se_path = openvoice_dir / "checkpoints_v2" / "base_speakers" / "ses" / "es.pth"

# =========================
# Leer prompts
# =========================
prompts = []
with open(prompts_csv, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompts.append({
            "prompt_id": row["prompt_id"],
            "texto": row["texto"]
        })

print(f"Prompts cargados: {len(prompts)}")
print(f"Speakers: {len(speakers)}")

# =========================
# Inicializar modelos una sola vez
# =========================
print("Cargando MeloTTS ES...")
melo_model = TTS(language="ES", device=device)
melo_speaker_id = melo_model.hps.data.spk2id["ES"]

print("Cargando OpenVoice ToneColorConverter...")
tone_color_converter = ToneColorConverter(
    str(ckpt_converter / "config.json"),
    device=device
)
tone_color_converter.load_ckpt(str(ckpt_converter / "checkpoint.pth"))

print("Cargando embedding fuente español...")
source_se = torch.load(str(source_se_path), map_location=device)

# =========================
# Preparar log
# =========================
if not log_csv.exists():
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "speaker_id",
            "prompt_id",
            "texto",
            "archivo_base_intermedio",
            "archivo_salida",
            "referencia_usada",
            "estado",
            "duracion_segundos",
            "error",
            "observaciones"
        ])

# =========================
# Extraer timbre por speaker una vez
# =========================
target_embeddings = {}

for spk in speakers:
    ref_audio = project_dir / "referencias_tts" / "openvoice" / f"{spk}_reference_openvoice.wav"

    print(f"\nExtrayendo timbre para {spk}")
    print("Referencia:", ref_audio)

    if not ref_audio.exists():
        raise FileNotFoundError(f"No existe la referencia de OpenVoice: {ref_audio}")

    target_se, audio_name = se_extractor.get_se(
        str(ref_audio),
        tone_color_converter,
        vad=True
    )

    target_embeddings[spk] = {
        "target_se": target_se,
        "ref_audio": ref_audio
    }

# =========================
# Generación completa
# =========================
total = len(speakers) * len(prompts)
contador = 0

for spk in speakers:
    (output_root / spk).mkdir(parents=True, exist_ok=True)
    (base_root / spk).mkdir(parents=True, exist_ok=True)

    target_se = target_embeddings[spk]["target_se"]
    ref_audio = target_embeddings[spk]["ref_audio"]

    for prompt in prompts:
        contador += 1

        prompt_id = prompt["prompt_id"]
        texto = prompt["texto"]

        base_audio = base_root / spk / f"{spk}_openvoice_base_{prompt_id}.wav"
        final_audio = output_root / spk / f"{spk}_openvoice_{prompt_id}.wav"

        print(f"\n[{contador}/{total}] {spk} - {prompt_id}")
        print("Texto:", texto)
        print("Base:", base_audio)
        print("Final:", final_audio)

        inicio = time.time()
        estado = ""
        error = ""

        try:
            if final_audio.exists() and final_audio.stat().st_size > 0:
                print("Final ya existe, se omite.")
                estado = "omitido_existente"
            else:
                # 1. Generar voz base si no existe
                if not (base_audio.exists() and base_audio.stat().st_size > 0):
                    print("Generando base MeloTTS...")
                    melo_model.tts_to_file(
                        texto,
                        melo_speaker_id,
                        str(base_audio),
                        speed=1.0
                    )
                else:
                    print("Base ya existe.")

                # 2. Convertir timbre con OpenVoice
                print("Convirtiendo con OpenVoice...")
                tone_color_converter.convert(
                    audio_src_path=str(base_audio),
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=str(final_audio),
                    message="ProyectoElectrico"
                )

                if final_audio.exists() and final_audio.stat().st_size > 0:
                    estado = "generado"
                else:
                    estado = "error"
                    error = "No se creó el archivo final."

            duracion = round(time.time() - inicio, 3)

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
                str(base_audio.relative_to(project_dir)).replace("\\", "/"),
                str(final_audio.relative_to(project_dir)).replace("\\", "/"),
                str(ref_audio.relative_to(project_dir)).replace("\\", "/"),
                estado,
                duracion,
                error,
                "OpenVoice V2 con base MeloTTS ES; audio final conserva rasgos de acento/prosodia de la voz base."
            ])

print("\nProceso OpenVoice finalizado.")
print("Log:", log_csv)
