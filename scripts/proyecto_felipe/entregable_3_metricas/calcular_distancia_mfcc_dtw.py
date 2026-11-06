import os
import argparse

import numpy as np
import pandas as pd
import librosa
from scipy.spatial.distance import cdist


TARGET_SR = 16000

SYSTEMS = {
    "xtts": "xtts_path",
    "f5tts": "f5tts_path",
    "openvoice": "openvoice_path",
    "elevenlabs": "elevenlabs_path",
}


def cargar_audio(path):
    # Librosa remuestrea en memoria a 16 kHz.
    y, sr = librosa.load(
        path,
        sr=TARGET_SR,
        mono=True
    )

    if len(y) == 0:
        raise ValueError("Audio vacío.")

    duracion_antes_trim = len(y) / sr

    # Eliminar silencios iniciales/finales.
    y_trim, _ = librosa.effects.trim(
        y,
        top_db=35
    )

    if len(y_trim) == 0:
        raise ValueError(
            "Audio vacío después de eliminar silencios."
        )

    duracion_despues_trim = len(y_trim) / sr

    return (
        y_trim,
        sr,
        duracion_antes_trim,
        duracion_despues_trim
    )


def calcular_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=1024,
        hop_length=256,
        n_mels=40,
        fmin=20,
        fmax=7600
    )

    # Excluir MFCC0 para disminuir el efecto
    # de diferencias globales de energía.
    return mfcc[1:13, :]


def distancia_mfcc_dtw(mfcc_a, mfcc_b):
    # librosa.sequence.dtw espera features x frames.
    D, wp = librosa.sequence.dtw(
        X=mfcc_a,
        Y=mfcc_b,
        metric="euclidean"
    )

    # wp contiene pares [frame_a, frame_b].
    # Calculamos la distancia euclídea de cada
    # par alineado y después su promedio.
    a = mfcc_a[:, wp[:, 0]].T
    b = mfcc_b[:, wp[:, 1]].T

    distancias = np.linalg.norm(
        a - b,
        axis=1
    )

    return {
        "distancia_media": float(np.mean(distancias)),
        "distancia_mediana": float(np.median(distancias)),
        "distancia_std": float(np.std(distancias, ddof=1))
        if len(distancias) > 1 else 0.0,
        "longitud_camino_dtw": int(len(wp)),
        "costo_dtw_acumulado": float(D[-1, -1]),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        required=True
    )

    parser.add_argument(
        "--manifest",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--summary",
        required=True
    )

    args = parser.parse_args()

    root = os.path.abspath(args.project_root)

    df = pd.read_csv(args.manifest)

    if len(df) != 125:
        raise RuntimeError(
            f"Se esperaban 125 filas y se encontraron {len(df)}."
        )

    os.makedirs(
        os.path.dirname(args.output),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(args.summary),
        exist_ok=True
    )

    print()
    print("============================================================")
    print(" ENTREGABLE 3 - DISTANCIA CEPSTRAL MFCC-DTW")
    print("============================================================")
    print()
    print("Filas del manifest:", len(df))
    print("Sample rate de análisis:", TARGET_SR, "Hz")
    print("MFCC usados: 1-12")
    print()

    resultados = []

    total = len(df)

    for indice, row in df.iterrows():

        numero = indice + 1

        sample_id = row["sample_id"]
        speaker_id = row["speaker_id"]
        prompt_id = row["prompt_id"]

        print(
            f"[{numero:03d}/{total}] "
            f"{sample_id}"
        )

        natural_rel = row["natural_path"]
        natural_abs = os.path.join(
            root,
            natural_rel
        )

        try:
            (
                y_nat,
                sr_nat,
                dur_nat_original,
                dur_nat_trim
            ) = cargar_audio(
                natural_abs
            )

            mfcc_nat = calcular_mfcc(
                y_nat,
                sr_nat
            )

            error_natural = None

        except Exception as exc:
            mfcc_nat = None
            dur_nat_original = None
            dur_nat_trim = None
            error_natural = str(exc)

            print(
                "   ERROR natural:",
                error_natural
            )

        for sistema, columna in SYSTEMS.items():

            synthetic_rel = row[columna]
            synthetic_abs = os.path.join(
                root,
                synthetic_rel
            )

            resultado = {
                "sample_id": sample_id,
                "speaker_id": speaker_id,
                "prompt_id": prompt_id,
                "sistema_tts": sistema,

                "natural_path": natural_rel,
                "synthetic_path": synthetic_rel,

                "sample_rate_analisis": TARGET_SR,

                "duracion_natural_original_s":
                    dur_nat_original,

                "duracion_natural_trim_s":
                    dur_nat_trim,

                "duracion_sintetica_original_s":
                    None,

                "duracion_sintetica_trim_s":
                    None,

                "distancia_mfcc_dtw_media":
                    None,

                "distancia_mfcc_dtw_mediana":
                    None,

                "distancia_mfcc_dtw_std":
                    None,

                "longitud_camino_dtw":
                    None,

                "costo_dtw_acumulado":
                    None,

                "estado": "error",
                "error": None,
            }

            if mfcc_nat is None:

                resultado["error"] = (
                    "Error en natural: "
                    + str(error_natural)
                )

                resultados.append(
                    resultado
                )

                continue

            try:
                (
                    y_syn,
                    sr_syn,
                    dur_syn_original,
                    dur_syn_trim
                ) = cargar_audio(
                    synthetic_abs
                )

                mfcc_syn = calcular_mfcc(
                    y_syn,
                    sr_syn
                )

                dist = distancia_mfcc_dtw(
                    mfcc_nat,
                    mfcc_syn
                )

                resultado[
                    "duracion_sintetica_original_s"
                ] = dur_syn_original

                resultado[
                    "duracion_sintetica_trim_s"
                ] = dur_syn_trim

                resultado[
                    "distancia_mfcc_dtw_media"
                ] = dist["distancia_media"]

                resultado[
                    "distancia_mfcc_dtw_mediana"
                ] = dist["distancia_mediana"]

                resultado[
                    "distancia_mfcc_dtw_std"
                ] = dist["distancia_std"]

                resultado[
                    "longitud_camino_dtw"
                ] = dist["longitud_camino_dtw"]

                resultado[
                    "costo_dtw_acumulado"
                ] = dist["costo_dtw_acumulado"]

                resultado["estado"] = "ok"

                print(
                    f"   {sistema:<11} "
                    f"{dist['distancia_media']:.4f}"
                )

            except Exception as exc:

                resultado["error"] = str(exc)

                print(
                    f"   {sistema:<11} "
                    f"ERROR: {exc}"
                )

            resultados.append(
                resultado
            )

        # Guardado incremental.
        if numero % 5 == 0:
            pd.DataFrame(
                resultados
            ).to_csv(
                args.output,
                index=False
            )

    resultados_df = pd.DataFrame(
        resultados
    )

    resultados_df.to_csv(
        args.output,
        index=False
    )

    correctos = resultados_df[
        resultados_df["estado"] == "ok"
    ].copy()

    resumen = (
        correctos
        .groupby("sistema_tts")
        .agg(
            n=(
                "distancia_mfcc_dtw_media",
                "count"
            ),
            promedio=(
                "distancia_mfcc_dtw_media",
                "mean"
            ),
            desviacion_estandar=(
                "distancia_mfcc_dtw_media",
                "std"
            ),
            mediana=(
                "distancia_mfcc_dtw_media",
                "median"
            ),
            minimo=(
                "distancia_mfcc_dtw_media",
                "min"
            ),
            maximo=(
                "distancia_mfcc_dtw_media",
                "max"
            ),
        )
        .reset_index()
    )

    resumen.to_csv(
        args.summary,
        index=False
    )

    print()
    print("============================================================")
    print(" PROCESAMIENTO FINALIZADO")
    print("============================================================")
    print()

    print(
        "Comparaciones generadas:",
        len(resultados_df)
    )

    print(
        "Comparaciones correctas:",
        (resultados_df["estado"] == "ok").sum()
    )

    print(
        "Comparaciones con error:",
        (resultados_df["estado"] != "ok").sum()
    )

    print()
    print(
        "Interpretación: menor distancia = "
        "mayor semejanza espectral."
    )

    print()
    print("Resumen:")
    print(
        resumen.to_string(
            index=False
        )
    )

    print()
    print("Resultados:")
    print(args.output)

    print()
    print("Resumen:")
    print(args.summary)

    print()


if __name__ == "__main__":
    main()
