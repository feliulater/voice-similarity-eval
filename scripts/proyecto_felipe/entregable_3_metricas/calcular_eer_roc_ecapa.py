import os
import argparse

import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt

from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics import roc_curve, auc


TARGET_SR = 16000

SYSTEMS = {
    "xtts": "xtts_path",
    "f5tts": "f5tts_path",
    "openvoice": "openvoice_path",
    "elevenlabs": "elevenlabs_path",
}

LABELS = {
    "xtts": "XTTS",
    "f5tts": "F5-TTS",
    "openvoice": "OpenVoice",
    "elevenlabs": "ElevenLabs",
}


def cargar_audio(path, device):
    wav, sr = torchaudio.load(path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(
            wav,
            orig_freq=sr,
            new_freq=TARGET_SR,
        )

    return wav.to(device)


def obtener_embedding(classifier, wav):
    with torch.no_grad():
        emb = classifier.encode_batch(wav)

    return emb.reshape(-1).cpu()


def cosine(a, b):
    return F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
        dim=1,
    ).item()


def calcular_eer(fpr, tpr, thresholds):
    fnr = 1.0 - tpr
    diferencia = fpr - fnr

    cruces = np.where(
        np.sign(diferencia[:-1]) != np.sign(diferencia[1:])
    )[0]

    if len(cruces) == 0:
        idx = np.argmin(np.abs(diferencia))
        eer = (fpr[idx] + fnr[idx]) / 2.0
        threshold = thresholds[idx]
        return eer, threshold

    i = cruces[0]

    d0 = diferencia[i]
    d1 = diferencia[i + 1]

    if d1 == d0:
        alpha = 0.0
    else:
        alpha = -d0 / (d1 - d0)

    eer = fpr[i] + alpha * (fpr[i + 1] - fpr[i])

    threshold = (
        thresholds[i]
        + alpha * (thresholds[i + 1] - thresholds[i])
    )

    return float(eer), float(threshold)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairs-output", required=True)
    parser.add_argument("--eer-output", required=True)
    parser.add_argument("--roc-output", required=True)

    args = parser.parse_args()

    root = os.path.abspath(args.project_root)
    df = pd.read_csv(args.manifest)

    if len(df) != 125:
        raise RuntimeError(
            f"Se esperaban 125 filas y hay {len(df)}."
        )

    os.makedirs(
        os.path.dirname(args.pairs_output),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(args.eer_output),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(args.roc_output),
        exist_ok=True
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print()
    print("============================================================")
    print(" ENTREGABLE 3 - EER / ROC CON ECAPA-TDNN")
    print("============================================================")
    print()
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

    # ========================================================
    # EMBEDDINGS NATURALES
    # ========================================================

    natural_embeddings = {}

    print("Calculando embeddings naturales...")

    for i, row in df.iterrows():
        key = (row["speaker_id"], row["prompt_id"])

        path = os.path.join(
            root,
            row["natural_path"]
        )

        wav = cargar_audio(path, device)

        natural_embeddings[key] = obtener_embedding(
            classifier,
            wav
        )

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/125")

    print("Embeddings naturales completos.")
    print()

    pares = []
    resumen_eer = []

    plt.figure(figsize=(8, 7))

    # ========================================================
    # PROCESAMIENTO POR SISTEMA
    # ========================================================

    for sistema, columna in SYSTEMS.items():

        print("------------------------------------------------------------")
        print("Sistema:", LABELS[sistema])
        print("------------------------------------------------------------")

        synthetic_embeddings = {}

        print("Calculando embeddings sintéticos...")

        for i, row in df.iterrows():

            key = (
                row["speaker_id"],
                row["prompt_id"]
            )

            path = os.path.join(
                root,
                row[columna]
            )

            wav = cargar_audio(
                path,
                device
            )

            synthetic_embeddings[key] = obtener_embedding(
                classifier,
                wav
            )

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/125")

        y_true = []
        scores = []

        # ====================================================
        # CONSTRUCCION DE PARES
        # ====================================================

        for _, row in df.iterrows():

            speaker_syn = row["speaker_id"]
            prompt = row["prompt_id"]

            emb_syn = synthetic_embeddings[
                (speaker_syn, prompt)
            ]

            # -----------------------------------------------
            # POSITIVO:
            # mismo speaker + mismo prompt
            # -----------------------------------------------

            emb_nat_pos = natural_embeddings[
                (speaker_syn, prompt)
            ]

            score_pos = cosine(
                emb_syn,
                emb_nat_pos
            )

            pares.append({
                "sistema_tts": sistema,
                "prompt_id": prompt,
                "speaker_sintetico": speaker_syn,
                "speaker_natural": speaker_syn,
                "tipo_par": "positivo",
                "label": 1,
                "similitud_coseno": score_pos,
            })

            y_true.append(1)
            scores.append(score_pos)

            # -----------------------------------------------
            # NEGATIVOS:
            # mismo prompt + otros speakers
            # -----------------------------------------------

            speakers_negativos = (
                df.loc[
                    df["prompt_id"] == prompt,
                    "speaker_id"
                ]
                .unique()
                .tolist()
            )

            for speaker_nat in speakers_negativos:

                if speaker_nat == speaker_syn:
                    continue

                emb_nat_neg = natural_embeddings[
                    (speaker_nat, prompt)
                ]

                score_neg = cosine(
                    emb_syn,
                    emb_nat_neg
                )

                pares.append({
                    "sistema_tts": sistema,
                    "prompt_id": prompt,
                    "speaker_sintetico": speaker_syn,
                    "speaker_natural": speaker_nat,
                    "tipo_par": "negativo",
                    "label": 0,
                    "similitud_coseno": score_neg,
                })

                y_true.append(0)
                scores.append(score_neg)

        y_true = np.asarray(y_true)
        scores = np.asarray(scores)

        # ====================================================
        # ROC, AUC Y EER
        # ====================================================

        fpr, tpr, thresholds = roc_curve(
            y_true,
            scores,
            pos_label=1
        )

        roc_auc = auc(fpr, tpr)

        eer, threshold_eer = calcular_eer(
            fpr,
            tpr,
            thresholds
        )

        positivos = scores[y_true == 1]
        negativos = scores[y_true == 0]

        resumen_eer.append({
            "sistema_tts": sistema,
            "n_positivos": len(positivos),
            "n_negativos": len(negativos),
            "eer": eer,
            "eer_porcentaje": eer * 100.0,
            "threshold_eer": threshold_eer,
            "auc": roc_auc,
            "media_positivos": positivos.mean(),
            "desv_positivos": positivos.std(ddof=1),
            "media_negativos": negativos.mean(),
            "desv_negativos": negativos.std(ddof=1),
        })

        print()
        print("Pares positivos:", len(positivos))
        print("Pares negativos:", len(negativos))
        print(f"EER: {eer:.6f} ({eer * 100:.2f} %)")
        print(f"Umbral EER: {threshold_eer:.6f}")
        print(f"AUC: {roc_auc:.6f}")
        print(
            "Media positivos:",
            f"{positivos.mean():.6f}"
        )
        print(
            "Media negativos:",
            f"{negativos.mean():.6f}"
        )
        print()

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                f"{LABELS[sistema]} "
                f"(AUC={roc_auc:.3f}, "
                f"EER={eer*100:.1f}%)"
            )
        )

    # ========================================================
    # GUARDAR PARES Y RESUMEN
    # ========================================================

    pares_df = pd.DataFrame(pares)

    pares_df.to_csv(
        args.pairs_output,
        index=False
    )

    resumen_df = pd.DataFrame(resumen_eer)

    resumen_df.to_csv(
        args.eer_output,
        index=False
    )

    # ========================================================
    # GRAFICA ROC
    # ========================================================

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Azar"
    )

    plt.xlabel("Tasa de falsos positivos (FPR)")
    plt.ylabel("Tasa de verdaderos positivos (TPR)")
    plt.title("Curvas ROC de verificación de hablante")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        args.roc_output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("============================================================")
    print(" PROCESAMIENTO FINALIZADO")
    print("============================================================")
    print()

    print("Pares totales:", len(pares_df))
    print()

    print("Pares por sistema:")
    print(
        pares_df.groupby(
            ["sistema_tts", "tipo_par"]
        ).size()
    )

    print()
    print("Resumen EER / ROC:")
    print(
        resumen_df.to_string(
            index=False
        )
    )

    print()
    print("Archivos generados:")
    print(args.pairs_output)
    print(args.eer_output)
    print(args.roc_output)
    print()


if __name__ == "__main__":
    main()
