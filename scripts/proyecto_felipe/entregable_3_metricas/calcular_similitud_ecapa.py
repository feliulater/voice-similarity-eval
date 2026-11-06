import os
import argparse

import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F

from speechbrain.inference.speaker import EncoderClassifier


TARGET_SR = 16000

SYSTEMS = {
    "xtts": "xtts_path",
    "f5tts": "f5tts_path",
    "openvoice": "openvoice_path",
    "elevenlabs": "elevenlabs_path",
}


def cargar_audio(path, device):
    wav, sr_original = torchaudio.load(path)

    # Convertir a mono si fuera necesario
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Duración real antes de cualquier remuestreo
    duracion = wav.shape[1] / sr_original

    remuestreado = False

    if sr_original != TARGET_SR:
        wav = torchaudio.functional.resample(
            wav,
            orig_freq=sr_original,
            new_freq=TARGET_SR,
        )
        remuestreado = True

    return (
        wav.to(device),
        sr_original,
        TARGET_SR,
        duracion,
        remuestreado,
    )


def obtener_embedding(classifier, wav):
    with torch.no_grad():
        return classifier.encode_batch(wav)


def similitud_coseno(emb_a, emb_b):
    return F.cosine_similarity(
        emb_a.reshape(1, -1),
        emb_b.reshape(1, -1),
        dim=1,
    ).item()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)

    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    manifest_path = os.path.abspath(args.manifest)
    output_path = os.path.abspath(args.output)
    summary_path = os.path.abspath(args.summary)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    print()
    print("============================================================")
    print(" ENTREGABLE 3 - SIMILITUD DE HABLANTE ECAPA-TDNN")
    print("============================================================")
    print()

    df = pd.read_csv(manifest_path)

    print("Filas del manifest:", len(df))

    if len(df) != 125:
        raise RuntimeError(
            f"Se esperaban 125 filas y se encontraron {len(df)}."
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Dispositivo:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("Cargando ECAPA-TDNN...")

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser(
            "~/ProyectoElectrico/pretrained_models/spkrec-ecapa-voxceleb"
        ),
        run_opts={"device": device},
    )

    print("Modelo cargado.")
    print()

    resultados = []

    for indice, row in df.iterrows():

        numero = indice + 1
        sample_id = row["sample_id"]
        speaker_id = row["speaker_id"]
        prompt_id = row["prompt_id"]

        print(f"[{numero:03d}/125] {sample_id}")

        natural_rel = row["natural_path"]
        natural_abs = os.path.join(project_root, natural_rel)

        try:
            (
                wav_nat,
                sr_nat_original,
                sr_nat_modelo,
                dur_nat,
                nat_remuestreado,
            ) = cargar_audio(natural_abs, device)

            emb_nat = obtener_embedding(classifier, wav_nat)
            natural_error = None

        except Exception as exc:
            emb_nat = None
            sr_nat_original = None
            sr_nat_modelo = None
            dur_nat = None
            nat_remuestreado = None
            natural_error = str(exc)

            print("   ERROR natural:", natural_error)

        for sistema, columna in SYSTEMS.items():

            synthetic_rel = row[columna]
            synthetic_abs = os.path.join(
                project_root,
                synthetic_rel,
            )

            resultado = {
                "sample_id": sample_id,
                "speaker_id": speaker_id,
                "prompt_id": prompt_id,
                "sistema_tts": sistema,

                "natural_path": natural_rel,
                "synthetic_path": synthetic_rel,

                "sample_rate_original_natural": sr_nat_original,
                "sample_rate_original_sintetico": None,

                "sample_rate_modelo": TARGET_SR,

                "remuestreado_natural": nat_remuestreado,
                "remuestreado_sintetico": None,

                "duracion_natural_s": dur_nat,
                "duracion_sintetica_s": None,

                "similitud_coseno": None,

                "estado": "error",
                "error": None,
            }

            if emb_nat is None:
                resultado["error"] = (
                    "Error en audio natural: "
                    + str(natural_error)
                )
                resultados.append(resultado)
                continue

            try:
                (
                    wav_syn,
                    sr_syn_original,
                    sr_syn_modelo,
                    dur_syn,
                    syn_remuestreado,
                ) = cargar_audio(
                    synthetic_abs,
                    device,
                )

                emb_syn = obtener_embedding(
                    classifier,
                    wav_syn,
                )

                sim = similitud_coseno(
                    emb_nat,
                    emb_syn,
                )

                resultado["sample_rate_original_sintetico"] = (
                    sr_syn_original
                )
                resultado["remuestreado_sintetico"] = (
                    syn_remuestreado
                )
                resultado["duracion_sintetica_s"] = dur_syn
                resultado["similitud_coseno"] = sim
                resultado["estado"] = "ok"

                marca = " [R]" if syn_remuestreado else ""

                print(
                    f"   {sistema:<11} "
                    f"{sim:.6f}{marca}"
                )

            except Exception as exc:

                resultado["error"] = str(exc)

                print(
                    f"   {sistema:<11} ERROR: {exc}"
                )

            resultados.append(resultado)

        # Guardado incremental
        if numero % 5 == 0:
            pd.DataFrame(resultados).to_csv(
                output_path,
                index=False,
            )

    resultados_df = pd.DataFrame(resultados)

    resultados_df.to_csv(
        output_path,
        index=False,
    )

    correctos = resultados_df[
        resultados_df["estado"] == "ok"
    ].copy()

    resumen = (
        correctos
        .groupby("sistema_tts")
        .agg(
            n=("similitud_coseno", "count"),
            promedio=("similitud_coseno", "mean"),
            desviacion_estandar=("similitud_coseno", "std"),
            mediana=("similitud_coseno", "median"),
            minimo=("similitud_coseno", "min"),
            maximo=("similitud_coseno", "max"),
            duracion_promedio_s=(
                "duracion_sintetica_s",
                "mean",
            ),
        )
        .reset_index()
    )

    resumen.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("============================================================")
    print(" PROCESAMIENTO FINALIZADO")
    print("============================================================")
    print()

    print("Comparaciones generadas:", len(resultados_df))
    print(
        "Comparaciones correctas:",
        (resultados_df["estado"] == "ok").sum(),
    )
    print(
        "Comparaciones con error:",
        (resultados_df["estado"] != "ok").sum(),
    )

    print()
    print("Audios sintéticos remuestreados:")
    print(
        int(
            resultados_df[
                "remuestreado_sintetico"
            ].fillna(False).sum()
        )
    )

    print()
    print("Resumen:")
    print(resumen.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
