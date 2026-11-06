from pathlib import Path
import os
import csv
import subprocess
import tempfile

import numpy as np
import soundfile as sf

project_dir = Path(
    os.environ.get(
        "PROYECTO_TTS_E2_DIR",
        Path.home() / "Proyecto_Electrico_Voces_TTS" / "entregable_2_voces",
    )
)

input_root = project_dir / "voces_sinteticas" / "xtts_raw"
output_root = project_dir / "voces_sinteticas" / "xtts"
metadata_dir = project_dir / "metadata"
log_csv = metadata_dir / "log_postprocesamiento_xtts.csv"

speakers = ["spk001", "spk002", "spk003", "spk004", "spk005"]

output_root.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)

def get_active_regions(audio, sr):
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    max_abs = np.max(np.abs(audio)) if len(audio) else 0
    if max_abs == 0:
        return [], audio

    frame_len = int(sr * 0.025)
    hop = int(sr * 0.010)

    rms = []
    positions = []

    for start in range(0, max(1, len(audio) - frame_len), hop):
        frame = audio[start:start + frame_len]
        if len(frame) == 0:
            continue
        rms.append(np.sqrt(np.mean(frame ** 2)))
        positions.append(start)

    rms = np.array(rms)
    positions = np.array(positions)

    if len(rms) == 0:
        return [], audio

    threshold = max(np.max(rms) * 0.04, 1e-4)
    active = rms > threshold

    regions = []
    in_region = False
    start_sample = 0

    for i, is_active in enumerate(active):
        if is_active and not in_region:
            in_region = True
            start_sample = positions[i]
        elif not is_active and in_region:
            in_region = False
            end_sample = positions[i] + frame_len
            regions.append([start_sample, min(end_sample, len(audio))])

    if in_region:
        regions.append([start_sample, len(audio)])

    # Unir regiones separadas por silencios muy cortos
    merged = []
    max_gap = int(sr * 0.35)

    for reg in regions:
        if not merged:
            merged.append(reg)
        else:
            gap = reg[0] - merged[-1][1]
            if gap <= max_gap:
                merged[-1][1] = reg[1]
            else:
                merged.append(reg)

    return merged, audio

def clean_audio(infile, outfile):
    audio, sr = sf.read(infile)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    regions, audio = get_active_regions(audio, sr)

    action = "copiado"
    cut_sample = len(audio)

    if len(regions) == 0:
        cleaned = audio
        action = "sin_actividad_detectada"

    else:
        # Si hay una última región muy corta después de un silencio largo,
        # se interpreta como posible artefacto aislado de cierre.
        if len(regions) >= 2:
            prev_start, prev_end = regions[-2]
            last_start, last_end = regions[-1]

            last_duration = (last_end - last_start) / sr
            gap_duration = (last_start - prev_end) / sr

            if last_duration <= 0.55 and gap_duration >= 0.50:
                cut_sample = min(len(audio), prev_end + int(0.25 * sr))
                action = "artefacto_final_recortado"
            else:
                cut_sample = min(len(audio), regions[-1][1] + int(0.25 * sr))
                action = "trim_final"
        else:
            cut_sample = min(len(audio), regions[-1][1] + int(0.25 * sr))
            action = "trim_final"

        cleaned = audio[:cut_sample]

    # Fade-out corto para evitar clicks
    fade_len = min(int(sr * 0.08), len(cleaned))
    if fade_len > 0:
        fade = np.linspace(1.0, 0.0, fade_len)
        cleaned[-fade_len:] *= fade

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    sf.write(tmp_path, cleaned, sr, subtype="PCM_16")

    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Estandarizar salida limpia a WAV mono 16 kHz, PCM 16-bit
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(tmp_path),
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        str(outfile)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    tmp_path.unlink(missing_ok=True)

    original_duration = len(audio) / sr
    cleaned_duration = cut_sample / sr

    return action, round(original_duration, 3), round(cleaned_duration, 3)

with open(log_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "speaker_id",
        "archivo_entrada",
        "archivo_salida",
        "accion",
        "duracion_original_seg",
        "duracion_limpia_seg"
    ])

total = 0

for spk in speakers:
    in_dir = input_root / spk
    out_dir = output_root / spk
    out_dir.mkdir(parents=True, exist_ok=True)

    for infile in sorted(in_dir.glob("*.wav")):
        outfile = out_dir / infile.name

        action, dur_orig, dur_clean = clean_audio(infile, outfile)

        print(f"{spk} | {infile.name} | {action} | {dur_orig}s -> {dur_clean}s")

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                spk,
                str(infile.relative_to(project_dir)).replace("\\", "/"),
                str(outfile.relative_to(project_dir)).replace("\\", "/"),
                action,
                dur_orig,
                dur_clean
            ])

        total += 1

print(f"\nPostprocesamiento terminado. Archivos procesados: {total}")
print(f"Log: {log_csv}")
