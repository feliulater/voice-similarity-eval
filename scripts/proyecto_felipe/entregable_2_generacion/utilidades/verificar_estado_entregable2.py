from pathlib import Path
import os
import csv
from collections import Counter, defaultdict

project_dir = Path(
    os.environ.get(
        "PROYECTO_TTS_E2_DIR",
        Path.home() / "Proyecto_Electrico_Voces_TTS" / "entregable_2_voces",
    )
)
metadata_dir = project_dir / "metadata"

checks = {
    "audios_naturales_wav_16k": list((project_dir / "audios_wav_16k").glob("spk*/spk*_real_prompt*.wav")),
    "xtts_finales": list((project_dir / "voces_sinteticas" / "xtts").glob("spk*/spk*_xtts_prompt*.wav")),
    "f5tts_finales": list((project_dir / "voces_sinteticas" / "f5tts").glob("spk*/spk*_f5tts_prompt*.wav")),
    "openvoice_finales": list((project_dir / "voces_sinteticas" / "openvoice").glob("spk*/spk*_openvoice_prompt*.wav")),
    "openvoice_intermedios": list((project_dir / "voces_sinteticas" / "openvoice_intermedios").glob("spk*/spk*_openvoice_base_prompt*.wav")),
}

print("\n=== Conteo de archivos ===")
for name, files in checks.items():
    print(f"{name}: {len(files)}")

print("\n=== Conteo por speaker ===")
for name, files in checks.items():
    counts = Counter(p.parent.name for p in files)
    print(f"\n{name}")
    for spk in ["spk001", "spk002", "spk003", "spk004", "spk005"]:
        print(f"  {spk}: {counts.get(spk, 0)}")

print("\n=== Metadata específica ===")
metadata_files = [
    "metadata_audios_naturales.csv",
    "metadata_voces_sinteticas_xtts.csv",
    "metadata_voces_sinteticas_f5tts.csv",
    "metadata_voces_sinteticas_openvoice.csv",
    "metadata_referencias_tts.csv",
    "plan_generacion_sintetica.csv",
]

for filename in metadata_files:
    path = metadata_dir / filename
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        print(f"{filename}: {len(rows)} filas")
    else:
        print(f"{filename}: NO EXISTE")

plan_path = metadata_dir / "plan_generacion_sintetica.csv"
if plan_path.exists():
    with open(plan_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    resumen = defaultdict(Counter)
    for row in rows:
        resumen[row["sistema_tts"]][row["estado"]] += 1

    print("\n=== Plan de generación por sistema ===")
    for sistema in sorted(resumen):
        print(f"{sistema}: {dict(resumen[sistema])}")

print("\nVerificación terminada.")
