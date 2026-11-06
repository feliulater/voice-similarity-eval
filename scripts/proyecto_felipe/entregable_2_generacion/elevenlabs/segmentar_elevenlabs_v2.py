from pathlib import Path
import os
import subprocess
import csv
import re
import math

project_dir = Path(
    os.environ.get(
        "PROYECTO_TTS_E2_DIR",
        Path.home() / "Proyecto_Electrico_Voces_TTS" / "entregable_2_voces",
    )
)

raw_root = project_dir / "voces_sinteticas" / "elevenlabs_raw"
out_root = project_dir / "voces_sinteticas" / "elevenlabs"
metadata_dir = project_dir / "metadata"
prompts_csv = metadata_dir / "prompts_final_entregable2.csv"
log_csv = metadata_dir / "log_segmentacion_elevenlabs.csv"

out_root.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def get_duration(path):
    result = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path)
    ])
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return float(result.stdout.strip())

def detect_silences(path):
    # Más conservador que la versión anterior.
    # Busca pausas más reales entre frases y evita pausas internas pequeñas.
    result = run([
        "ffmpeg", "-hide_banner", "-i", str(path),
        "-af", "silencedetect=n=-30dB:d=0.15",
        "-f", "null", "-"
    ])

    text = result.stderr
    silences = []
    current_start = None

    for line in text.splitlines():
        if "silence_start:" in line:
            current_start = float(line.split("silence_start:")[1].strip())
        elif "silence_end:" in line and current_start is not None:
            m = re.search(r"silence_end: ([0-9.]+) \| silence_duration: ([0-9.]+)", line)
            if m:
                end = float(m.group(1))
                dur = float(m.group(2))
                mid = (current_start + end) / 2
                silences.append({
                    "start": current_start,
                    "end": end,
                    "duration": dur,
                    "mid": mid
                })
            current_start = None

    return silences

def cluster_silences(silences, min_gap=0.55):
    # Si hay varios silencios muy cercanos, conservar el más largo.
    if not silences:
        return []

    silences = sorted(silences, key=lambda s: s["mid"])
    clusters = [[silences[0]]]

    for s in silences[1:]:
        if s["mid"] - clusters[-1][-1]["mid"] < min_gap:
            clusters[-1].append(s)
        else:
            clusters.append([s])

    selected = []
    for cluster in clusters:
        best = max(cluster, key=lambda s: s["duration"])
        selected.append(best)

    return selected

def choose_cuts(duration, silences, prompts):
    candidates = [
        s for s in silences
        if s["mid"] > 0.40 and s["mid"] < duration - 0.40
    ]

    candidates = cluster_silences(candidates)

    print("Candidatos conservadores:", len(candidates))
    print("Tiempos candidatos:", [round(c["mid"], 2) for c in candidates])

    if len(candidates) < 24:
        raise RuntimeError(f"Solo se detectaron {len(candidates)} silencios candidatos; se necesitan al menos 24.")

    # Estimar posiciones esperadas según largo de texto.
    weights = [max(8, len(p["texto"])) for p in prompts]
    total_w = sum(weights)

    expected = []
    acc = 0
    for w in weights[:-1]:
        acc += w
        expected.append(duration * acc / total_w)

    n = len(candidates)
    m = 24
    min_segment = 1.15

    dp = [[math.inf] * n for _ in range(m)]
    prev = [[None] * n for _ in range(m)]

    for i, c in enumerate(candidates):
        t = c["mid"]
        if t >= min_segment:
            dp[0][i] = (t - expected[0]) ** 2 - 0.20 * c["duration"]

    for k in range(1, m):
        for i, c in enumerate(candidates):
            t = c["mid"]
            local_cost = (t - expected[k]) ** 2 - 0.20 * c["duration"]

            best_cost = math.inf
            best_j = None

            for j in range(i):
                prev_t = candidates[j]["mid"]
                if t - prev_t < min_segment:
                    continue

                cost = dp[k-1][j] + local_cost
                if cost < best_cost:
                    best_cost = cost
                    best_j = j

            dp[k][i] = best_cost
            prev[k][i] = best_j

    best_i = None
    best_cost = math.inf

    for i, c in enumerate(candidates):
        if duration - c["mid"] < min_segment:
            continue
        if dp[m-1][i] < best_cost:
            best_cost = dp[m-1][i]
            best_i = i

    if best_i is None:
        raise RuntimeError("No se pudo seleccionar una secuencia válida de 24 cortes.")

    selected = []
    k = m - 1
    i = best_i

    while k >= 0:
        selected.append(candidates[i])
        i = prev[k][i]
        k -= 1

    selected.reverse()

    print("Cortes seleccionados:", [round(s["mid"], 2) for s in selected])

    cuts = [0.0] + [s["mid"] for s in selected] + [duration]
    return cuts

def extract_segment(input_path, output_path, start, end):
    pad = 0.05
    seg_start = max(0.0, start - pad)
    seg_end = end + pad
    seg_dur = max(0.1, seg_end - seg_start)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ss", f"{seg_start:.3f}",
        "-t", f"{seg_dur:.3f}",
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        str(output_path)
    ]

    result = run(cmd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

def read_prompts():
    with open(prompts_csv, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

prompts = read_prompts()
if len(prompts) != 25:
    raise RuntimeError(f"Se esperaban 25 prompts, pero se encontraron {len(prompts)}.")

raw_files = sorted(raw_root.glob("spk*_elevenlabs_raw.mp3"))
print("Archivos raw encontrados:", len(raw_files))

with open(log_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "speaker_id",
        "prompt_id",
        "texto",
        "archivo_raw",
        "archivo_salida",
        "inicio_seg",
        "fin_seg",
        "duracion_seg",
        "estado",
        "observaciones"
    ])

for raw_path in raw_files:
    m = re.search(r"(spk\d{3})", raw_path.name)
    if not m:
        print("No se pudo identificar speaker en:", raw_path.name)
        continue

    speaker_id = m.group(1)
    speaker_out = out_root / speaker_id
    speaker_out.mkdir(parents=True, exist_ok=True)

    print(f"\nProcesando {speaker_id}")
    print("Archivo:", raw_path)

    duration = get_duration(raw_path)
    silences = detect_silences(raw_path)
    cuts = choose_cuts(duration, silences, prompts)

    print(f"Duración raw: {duration:.3f} s")

    for idx, prompt in enumerate(prompts):
        prompt_id = prompt["prompt_id"]
        texto = prompt["texto"]

        start = cuts[idx]
        end = cuts[idx + 1]
        out_path = speaker_out / f"{speaker_id}_elevenlabs_{prompt_id}.wav"

        estado = "generado"
        obs = "Segmentado desde salida MP3 larga de ElevenLabs; convertido a WAV mono 16 kHz."

        try:
            extract_segment(raw_path, out_path, start, end)
        except Exception as e:
            estado = "error"
            obs = repr(e)

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                speaker_id,
                prompt_id,
                texto,
                str(raw_path.relative_to(project_dir)).replace("\\", "/"),
                str(out_path.relative_to(project_dir)).replace("\\", "/"),
                round(start, 3),
                round(end, 3),
                round(end - start, 3),
                estado,
                obs
            ])

        print(f"{speaker_id} {prompt_id}: {estado} | {end-start:.3f}s")

print("\nSegmentación ElevenLabs terminada.")
print("Log:", log_csv)
