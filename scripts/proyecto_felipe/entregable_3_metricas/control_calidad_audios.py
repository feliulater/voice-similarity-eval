import os
import pandas as pd
import numpy as np
import soundfile as sf
import librosa


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
    "resultados_csv",
    "control_calidad_audios.csv"
)

RESUMEN = os.path.join(
    ROOT,
    "entregable_3_metricas",
    "tablas_resumen",
    "resumen_control_calidad_audios.csv"
)

df = pd.read_csv(MANIFEST)

fuentes = {
    "natural": "natural_path",
    "xtts": "xtts_path",
    "f5tts": "f5tts_path",
    "openvoice": "openvoice_path",
    "elevenlabs": "elevenlabs_path",
}

resultados = []

total = len(df) * len(fuentes)
contador = 0

print()
print("============================================================")
print(" CONTROL DE CALIDAD - 625 AUDIOS")
print("============================================================")
print()

for _, row in df.iterrows():

    for fuente, columna in fuentes.items():

        contador += 1

        ruta_rel = row[columna]
        ruta_abs = os.path.join(ROOT, ruta_rel)

        try:
            info = sf.info(ruta_abs)

            audio, sr = sf.read(
                ruta_abs,
                always_2d=False
            )

            audio = np.asarray(audio)

            if audio.ndim == 2:
                mono = audio.mean(axis=1)
                canales = audio.shape[1]
            else:
                mono = audio
                canales = 1

            duracion = len(mono) / sr

            # Trim únicamente para analizar duración útil.
            trim, _ = librosa.effects.trim(
                mono.astype(float),
                top_db=35
            )

            duracion_trim = len(trim) / sr

            silencio_extremos = (
                duracion - duracion_trim
            )

            if len(mono) > 0:
                pico = float(
                    np.max(np.abs(mono))
                )

                rms = float(
                    np.sqrt(
                        np.mean(
                            mono.astype(float) ** 2
                        )
                    )
                )

                proporcion_cerca_clipping = float(
                    np.mean(
                        np.abs(mono) >= 0.999
                    )
                )
            else:
                pico = np.nan
                rms = np.nan
                proporcion_cerca_clipping = np.nan

            resultados.append({
                "sample_id": row["sample_id"],
                "speaker_id": row["speaker_id"],
                "prompt_id": row["prompt_id"],
                "fuente": fuente,

                "ruta": ruta_rel,

                "sample_rate": sr,
                "canales": canales,
                "subtipo": info.subtype,

                "duracion_s": duracion,
                "duracion_trim_s": duracion_trim,
                "silencio_extremos_s": silencio_extremos,

                "pico_abs": pico,
                "rms": rms,
                "proporcion_cerca_clipping": proporcion_cerca_clipping,

                "estado": "ok",
                "error": None,
            })

        except Exception as exc:

            resultados.append({
                "sample_id": row["sample_id"],
                "speaker_id": row["speaker_id"],
                "prompt_id": row["prompt_id"],
                "fuente": fuente,

                "ruta": ruta_rel,

                "sample_rate": None,
                "canales": None,
                "subtipo": None,

                "duracion_s": None,
                "duracion_trim_s": None,
                "silencio_extremos_s": None,

                "pico_abs": None,
                "rms": None,
                "proporcion_cerca_clipping": None,

                "estado": "error",
                "error": str(exc),
            })

        if contador % 50 == 0:
            print(f"{contador}/{total}")

print(f"{total}/{total}")

out = pd.DataFrame(resultados)

out.to_csv(
    SALIDA,
    index=False
)

correctos = out[
    out["estado"] == "ok"
].copy()

resumen = (
    correctos
    .groupby("fuente")
    .agg(
        n=("sample_id", "count"),

        duracion_media_s=(
            "duracion_s",
            "mean"
        ),

        duracion_trim_media_s=(
            "duracion_trim_s",
            "mean"
        ),

        duracion_trim_std_s=(
            "duracion_trim_s",
            "std"
        ),

        silencio_extremos_medio_s=(
            "silencio_extremos_s",
            "mean"
        ),

        pico_medio=(
            "pico_abs",
            "mean"
        ),

        rms_medio=(
            "rms",
            "mean"
        ),

        clipping_maximo=(
            "proporcion_cerca_clipping",
            "max"
        ),
    )
    .reset_index()
)

resumen.to_csv(
    RESUMEN,
    index=False
)

print()
print("============================================================")
print(" RESULTADO")
print("============================================================")
print()

print("Audios analizados:", len(out))
print(
    "Correctos:",
    (out["estado"] == "ok").sum()
)
print(
    "Errores:",
    (out["estado"] != "ok").sum()
)

print()
print("Conteo por fuente:")
print(
    out["fuente"]
    .value_counts()
    .sort_index()
)

print()
print("Sample rates:")
print(
    correctos.groupby(
        ["fuente", "sample_rate"]
    ).size()
)

print()
print("Canales:")
print(
    correctos.groupby(
        ["fuente", "canales"]
    ).size()
)

print()
print("Resumen:")
print(
    resumen.to_string(
        index=False
    )
)

print()
print("CSV completo:")
print(SALIDA)

print()
print("Resumen:")
print(RESUMEN)
print()
