from pathlib import Path
import pandas as pd
import random
import shutil

# ============================================================
# CONFIGURACION
# ============================================================

ROOT = Path(
    os.environ.get(
        "PROYECTO_TTS_ROOT",
        Path.home() / "Proyecto_Electrico_Voces_TTS",
    )
)

E3 = ROOT / "entregable_3_metricas"
E4 = ROOT / "entregable_4_analisis_final"
PER = E4 / "evaluacion_perceptual"

MANIFEST = E3 / "config" / "manifest_evaluacion.csv"

CONFIG = PER / "config"
AUDIOS = PER / "audios"

VERSIONES = ["A", "B", "C", "D"]

SISTEMAS = [
    ("xtts", "xtts_path"),
    ("f5tts", "f5tts_path"),
    ("openvoice", "openvoice_path"),
    ("elevenlabs", "elevenlabs_path"),
]

SEED = 20261001

# ============================================================
# LEER Y VALIDAR MANIFEST
# ============================================================

df = pd.read_csv(MANIFEST)

columnas_necesarias = [
    "sample_id",
    "speaker_id",
    "prompt_id",
    "natural_path",
    "xtts_path",
    "f5tts_path",
    "openvoice_path",
    "elevenlabs_path",
]

faltantes = [
    c for c in columnas_necesarias
    if c not in df.columns
]

if faltantes:
    raise ValueError(
        f"Faltan columnas en el manifest: {faltantes}"
    )

speakers = sorted(df["speaker_id"].unique())

if len(speakers) != 5:
    raise ValueError(
        f"Se esperaban 5 hablantes y se encontraron {len(speakers)}"
    )

# ============================================================
# SELECCIONAR PROMPTS
#
# Para cada hablante:
#   4 prompts objetivo -> uno por version
#   4 prompts referencia -> uno por version
#
# Ninguna referencia usa el mismo prompt que el audio objetivo.
# ============================================================

rng = random.Random(SEED)

seleccion_bloques = []

for speaker in speakers:

    datos_spk = (
        df[df["speaker_id"] == speaker]
        .sort_values("prompt_id")
        .copy()
    )

    prompts = datos_spk["prompt_id"].tolist()

    if len(prompts) < 8:
        raise ValueError(
            f"{speaker} no tiene suficientes prompts."
        )

    rng.shuffle(prompts)

    prompts_objetivo = prompts[:4]
    prompts_referencia = prompts[4:8]

    for i, version in enumerate(VERSIONES):

        target_prompt = prompts_objetivo[i]
        reference_prompt = prompts_referencia[i]

        if target_prompt == reference_prompt:
            raise RuntimeError(
                "Prompt objetivo y referencia coinciden."
            )

        seleccion_bloques.append({
            "version": version,
            "speaker_id": speaker,
            "target_prompt_id": target_prompt,
            "reference_prompt_id": reference_prompt,
        })

bloques = pd.DataFrame(seleccion_bloques)

# ============================================================
# GENERAR ASIGNACION CIEGA A/B/C/D
#
# Latin square:
# cada sistema aparece equilibradamente en las posiciones
# A, B, C y D a lo largo del experimento.
# ============================================================

filas = []

labels = ["A", "B", "C", "D"]

for v_idx, version in enumerate(VERSIONES):

    bloques_version = bloques[
        bloques["version"] == version
    ].copy()

    for s_idx, (_, bloque) in enumerate(
        bloques_version.iterrows()
    ):

        speaker = bloque["speaker_id"]
        target_prompt = bloque["target_prompt_id"]
        reference_prompt = bloque["reference_prompt_id"]

        fila_target = df[
            (df["speaker_id"] == speaker) &
            (df["prompt_id"] == target_prompt)
        ]

        fila_ref = df[
            (df["speaker_id"] == speaker) &
            (df["prompt_id"] == reference_prompt)
        ]

        if len(fila_target) != 1:
            raise ValueError(
                f"No se encontro exactamente un target para "
                f"{speaker} {target_prompt}"
            )

        if len(fila_ref) != 1:
            raise ValueError(
                f"No se encontro exactamente una referencia para "
                f"{speaker} {reference_prompt}"
            )

        fila_target = fila_target.iloc[0]
        fila_ref = fila_ref.iloc[0]

        # Rotacion balanceada de sistemas
        shift = (v_idx + s_idx) % len(SISTEMAS)

        sistemas_ordenados = (
            SISTEMAS[shift:] +
            SISTEMAS[:shift]
        )

        # Nombre ciego del hablante dentro de la encuesta
        speaker_num = s_idx + 1
        speaker_label = f"Voz_{speaker_num}"

        # -------------------------
        # COPIAR REFERENCIA NATURAL
        # -------------------------

        ref_source = ROOT / str(fila_ref["natural_path"])

        ref_name = (
            f"{version}_{speaker_label}_REFERENCIA.wav"
        )

        ref_dest = (
            AUDIOS /
            f"version_{version}" /
            ref_name
        )

        if not ref_source.exists():
            raise FileNotFoundError(
                f"No existe referencia: {ref_source}"
            )

        shutil.copy2(
            ref_source,
            ref_dest
        )

        # -------------------------
        # COPIAR SISTEMAS CIEGOS
        # -------------------------

        for pos, (sistema, columna) in enumerate(
            sistemas_ordenados
        ):

            blind_label = labels[pos]

            source = ROOT / str(
                fila_target[columna]
            )

            audio_name = (
                f"{version}_{speaker_label}_"
                f"Audio_{blind_label}.wav"
            )

            dest = (
                AUDIOS /
                f"version_{version}" /
                audio_name
            )

            if not source.exists():
                raise FileNotFoundError(
                    f"No existe audio: {source}"
                )

            shutil.copy2(
                source,
                dest
            )

            filas.append({
                "version": version,
                "speaker_id": speaker,
                "speaker_label": speaker_label,
                "target_prompt_id": target_prompt,
                "reference_prompt_id": reference_prompt,
                "blind_label": blind_label,
                "sistema_tts": sistema,
                "reference_original_path":
                    str(fila_ref["natural_path"]),
                "synthetic_original_path":
                    str(fila_target[columna]),
                "reference_file": ref_name,
                "synthetic_file": audio_name,
            })

asignacion = pd.DataFrame(filas)

# ============================================================
# GUARDAR MAPA INTERNO
#
# ESTE ARCHIVO NO SE COMPARTE CON LOS EVALUADORES.
# ============================================================

archivo_interno = (
    CONFIG /
    "asignacion_perceptual_interna.csv"
)

asignacion.to_csv(
    archivo_interno,
    index=False
)

# ============================================================
# CREAR ARCHIVOS PARA CONSTRUIR CADA FORMULARIO
#
# Estos archivos NO contienen el nombre real del sistema.
# ============================================================

for version in VERSIONES:

    vis = asignacion[
        asignacion["version"] == version
    ][
        [
            "speaker_label",
            "target_prompt_id",
            "reference_prompt_id",
            "blind_label",
            "reference_file",
            "synthetic_file",
        ]
    ].copy()

    vis.to_csv(
        CONFIG /
        f"formulario_version_{version}.csv",
        index=False
    )

# ============================================================
# VALIDACIONES
# ============================================================

print()
print("============================================================")
print(" EVALUACION PERCEPTUAL PREPARADA")
print("============================================================")
print()

print("Filas sinteticas:", len(asignacion))

print()
print("Audios sinteticos por version:")
print(
    asignacion["version"]
    .value_counts()
    .sort_index()
)

print()
print("Audios sinteticos por sistema:")
print(
    asignacion["sistema_tts"]
    .value_counts()
    .sort_index()
)

print()
print("Audios sinteticos por hablante:")
print(
    asignacion["speaker_id"]
    .value_counts()
    .sort_index()
)

print()
print("Balance sistema x etiqueta ciega:")
print(
    pd.crosstab(
        asignacion["sistema_tts"],
        asignacion["blind_label"]
    )
)

print()
print("Prompts objetivo iguales a referencia:")
print(
    (
        asignacion["target_prompt_id"]
        ==
        asignacion["reference_prompt_id"]
    ).sum()
)

print()
print("Archivos fisicos por version:")

for version in VERSIONES:

    carpeta = AUDIOS / f"version_{version}"

    archivos = list(
        carpeta.glob("*.wav")
    )

    print(
        f"Version {version}: {len(archivos)}"
    )

print()
print("Mapa interno:")
print(archivo_interno)

print()
print("IMPORTANTE:")
print(
    "asignacion_perceptual_interna.csv contiene "
    "la identidad de los sistemas y NO debe compartirse "
    "con los participantes."
)
print()
