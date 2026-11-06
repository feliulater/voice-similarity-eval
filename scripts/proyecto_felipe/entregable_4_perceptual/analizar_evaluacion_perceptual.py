from pathlib import Path
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare, wilcoxon, spearmanr


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

PROCESADOS = PER / "resultados" / "procesados"
GRAFICAS = PER / "graficas"
TABLAS = PER / "tablas_resumen"

GRAFICAS.mkdir(parents=True, exist_ok=True)
TABLAS.mkdir(parents=True, exist_ok=True)

ARCHIVO_EVALUACIONES = (
    PROCESADOS / "evaluaciones_perceptuales_por_audio.csv"
)

ARCHIVO_OBJETIVAS = (
    E3 / "tablas_resumen" / "resumen_general_metricas_entregable3.csv"
)

ORDEN_SISTEMAS = [
    "xtts",
    "f5tts",
    "openvoice",
    "elevenlabs",
]

NOMBRES = {
    "xtts": "XTTS",
    "f5tts": "F5-TTS",
    "openvoice": "OpenVoice",
    "elevenlabs": "ElevenLabs",
}


# ============================================================
# FUNCIONES
# ============================================================

def normalizar_sistema(nombre):
    """
    Convierte distintas formas de escribir los sistemas
    a los nombres internos canonicos.
    """

    texto = str(nombre).strip().lower()

    texto = (
        texto
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    mapa = {
        "xtts": "xtts",
        "f5tts": "f5tts",
        "openvoice": "openvoice",
        "elevenlabs": "elevenlabs",
    }

    return mapa.get(texto, texto)


def correccion_holm(p_values):
    """
    Correccion de Holm para comparaciones multiples.
    """

    p_values = np.asarray(p_values, dtype=float)

    m = len(p_values)

    orden = np.argsort(p_values)

    ajustados = np.empty(m, dtype=float)

    max_previo = 0.0

    for posicion, indice in enumerate(orden):

        factor = m - posicion

        valor = factor * p_values[indice]

        valor = min(valor, 1.0)

        max_previo = max(max_previo, valor)

        ajustados[indice] = max_previo

    return ajustados


def guardar_boxplot(df, metrica, titulo, ylabel, nombre_archivo):

    datos = [
        df.loc[
            df["sistema_tts"] == sistema,
            metrica
        ].values
        for sistema in ORDEN_SISTEMAS
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.boxplot(
        datos,
        tick_labels=[
            NOMBRES[s]
            for s in ORDEN_SISTEMAS
        ],
        showmeans=True
    )

    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        GRAFICAS / nombre_archivo,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def guardar_barras_medias(
    resumen,
    metrica_media,
    metrica_ci,
    titulo,
    ylabel,
    nombre_archivo
):

    datos = (
        resumen
        .set_index("sistema_tts")
        .reindex(ORDEN_SISTEMAS)
    )

    medias = datos[metrica_media].values
    errores = datos[metrica_ci].values

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.bar(
        [NOMBRES[s] for s in ORDEN_SISTEMAS],
        medias,
        yerr=errores,
        capsize=5
    )

    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 5.5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        GRAFICAS / nombre_archivo,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def guardar_distribucion(
    distribucion,
    metrica,
    titulo,
    nombre_archivo
):

    subset = distribucion[
        distribucion["metrica"] == metrica
    ].copy()

    pivot = (
        subset
        .pivot(
            index="sistema_tts",
            columns="puntuacion",
            values="porcentaje"
        )
        .reindex(ORDEN_SISTEMAS)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bottom = np.zeros(len(pivot))

    x = np.arange(len(pivot))

    for puntuacion in [1, 2, 3, 4, 5]:

        valores = pivot[puntuacion].values

        ax.bar(
            x,
            valores,
            bottom=bottom,
            label=str(puntuacion)
        )

        bottom += valores

    ax.set_xticks(x)
    ax.set_xticklabels(
        [NOMBRES[s] for s in ORDEN_SISTEMAS]
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Porcentaje de respuestas (%)")
    ax.set_title(titulo)

    ax.legend(
        title="Puntuación",
        ncols=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12)
    )

    fig.tight_layout()

    fig.savefig(
        GRAFICAS / nombre_archivo,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def guardar_por_hablante(
    resumen,
    metrica,
    titulo,
    ylabel,
    nombre_archivo
):

    pivot = (
        resumen
        .pivot(
            index="speaker_id",
            columns="sistema_tts",
            values=metrica
        )
        .reindex(columns=ORDEN_SISTEMAS)
    )

    x = np.arange(len(pivot.index))

    ancho = 0.18

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, sistema in enumerate(ORDEN_SISTEMAS):

        desplazamiento = (
            i - (len(ORDEN_SISTEMAS) - 1) / 2
        ) * ancho

        ax.bar(
            x + desplazamiento,
            pivot[sistema].values,
            width=ancho,
            label=NOMBRES[sistema]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)

    ax.set_ylim(0, 5.5)

    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Hablante")

    ax.legend()

    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        GRAFICAS / nombre_archivo,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def guardar_scatter(
    df,
    x_col,
    y_col,
    xlabel,
    ylabel,
    titulo,
    nombre_archivo
):

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for _, fila in df.iterrows():

        sistema = fila["sistema_tts"]

        ax.scatter(
            fila[x_col],
            fila[y_col],
            s=80
        )

        ax.annotate(
            NOMBRES[sistema],
            (
                fila[x_col],
                fila[y_col]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)

    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        GRAFICAS / nombre_archivo,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# LEER DATOS PERCEPTUALES
# ============================================================

evaluaciones = pd.read_csv(
    ARCHIVO_EVALUACIONES,
    encoding="utf-8-sig"
)

evaluaciones["sistema_tts"] = (
    evaluaciones["sistema_tts"]
    .apply(normalizar_sistema)
)

print()
print("============================================================")
print(" ANALISIS ESTADISTICO DE EVALUACION PERCEPTUAL")
print("============================================================")
print()

print("Evaluaciones:", len(evaluaciones))
print("Participantes:", evaluaciones["participante_id"].nunique())
print("Sistemas:", evaluaciones["sistema_tts"].nunique())

if len(evaluaciones) != 400:
    raise ValueError(
        "La base perceptual no contiene 400 evaluaciones."
    )

if evaluaciones["participante_id"].nunique() != 20:
    raise ValueError(
        "Se esperaban 20 participantes."
    )


# ============================================================
# 1. RESUMEN GLOBAL POR SISTEMA
# ============================================================

resumen_global = (
    evaluaciones
    .groupby("sistema_tts")
    .agg(
        n=("similitud", "size"),

        similitud_media=("similitud", "mean"),
        similitud_std=("similitud", "std"),
        similitud_mediana=("similitud", "median"),

        naturalidad_media=("naturalidad", "mean"),
        naturalidad_std=("naturalidad", "std"),
        naturalidad_mediana=("naturalidad", "median"),
    )
    .reset_index()
)

resumen_global = (
    resumen_global
    .set_index("sistema_tts")
    .reindex(ORDEN_SISTEMAS)
    .reset_index()
)

resumen_global[
    resumen_global.select_dtypes(include="number").columns
] = (
    resumen_global
    .select_dtypes(include="number")
    .round(4)
)

resumen_global.to_csv(
    TABLAS / "resumen_perceptual_global.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 2. MEDIA POR PARTICIPANTE Y SISTEMA
#
# Esta tabla es la base correcta para las pruebas inferenciales.
# Cada participante aporta una media por sistema.
# ============================================================

por_participante = (
    evaluaciones
    .groupby(
        [
            "participante_id",
            "version",
            "sistema_tts",
        ]
    )
    .agg(
        n_audios=("similitud", "size"),
        similitud_media=("similitud", "mean"),
        naturalidad_media=("naturalidad", "mean"),
    )
    .reset_index()
)

if len(por_participante) != 80:
    raise ValueError(
        f"Se esperaban 80 filas participante-sistema "
        f"y se obtuvieron {len(por_participante)}."
    )

por_participante.to_csv(
    PROCESADOS /
    "resumen_perceptual_por_participante_sistema.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 3. IC 95 % DE LAS MEDIAS A NIVEL DE PARTICIPANTE
# ============================================================

resumen_participantes = (
    por_participante
    .groupby("sistema_tts")
    .agg(
        n_participantes=("participante_id", "size"),

        similitud_media=("similitud_media", "mean"),
        similitud_std=("similitud_media", "std"),

        naturalidad_media=("naturalidad_media", "mean"),
        naturalidad_std=("naturalidad_media", "std"),
    )
    .reset_index()
)

resumen_participantes["similitud_se"] = (
    resumen_participantes["similitud_std"]
    /
    np.sqrt(resumen_participantes["n_participantes"])
)

resumen_participantes["naturalidad_se"] = (
    resumen_participantes["naturalidad_std"]
    /
    np.sqrt(resumen_participantes["n_participantes"])
)

# Aproximacion normal 95 %
resumen_participantes["similitud_ci95"] = (
    1.96 * resumen_participantes["similitud_se"]
)

resumen_participantes["naturalidad_ci95"] = (
    1.96 * resumen_participantes["naturalidad_se"]
)

resumen_participantes = (
    resumen_participantes
    .set_index("sistema_tts")
    .reindex(ORDEN_SISTEMAS)
    .reset_index()
)

resumen_participantes[
    resumen_participantes
    .select_dtypes(include="number")
    .columns
] = (
    resumen_participantes
    .select_dtypes(include="number")
    .round(4)
)

resumen_participantes.to_csv(
    TABLAS /
    "resumen_perceptual_participantes_ic95.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 4. RESUMEN POR HABLANTE
# ============================================================

por_hablante = (
    evaluaciones
    .groupby(
        ["speaker_id", "sistema_tts"]
    )
    .agg(
        n=("similitud", "size"),
        similitud_media=("similitud", "mean"),
        similitud_std=("similitud", "std"),
        naturalidad_media=("naturalidad", "mean"),
        naturalidad_std=("naturalidad", "std"),
    )
    .reset_index()
)

por_hablante[
    por_hablante
    .select_dtypes(include="number")
    .columns
] = (
    por_hablante
    .select_dtypes(include="number")
    .round(4)
)

por_hablante.to_csv(
    TABLAS /
    "resumen_perceptual_por_hablante.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. DISTRIBUCION DE PUNTUACIONES 1-5
# ============================================================

filas_distribucion = []

for sistema in ORDEN_SISTEMAS:

    subset = evaluaciones[
        evaluaciones["sistema_tts"] == sistema
    ]

    for metrica in ["similitud", "naturalidad"]:

        conteos = (
            subset[metrica]
            .value_counts()
            .reindex(
                [1, 2, 3, 4, 5],
                fill_value=0
            )
        )

        total = conteos.sum()

        for puntuacion, cantidad in conteos.items():

            filas_distribucion.append({
                "sistema_tts": sistema,
                "metrica": metrica,
                "puntuacion": int(puntuacion),
                "cantidad": int(cantidad),
                "porcentaje": (
                    100 * cantidad / total
                ),
            })

distribucion = pd.DataFrame(
    filas_distribucion
)

distribucion["porcentaje"] = (
    distribucion["porcentaje"].round(2)
)

distribucion.to_csv(
    TABLAS /
    "distribucion_perceptual_1_a_5.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 6. PRUEBA DE FRIEDMAN
#
# Se trabaja sobre medias participante-sistema.
# ============================================================

resultados_friedman = []

for metrica in [
    "similitud_media",
    "naturalidad_media",
]:

    tabla = (
        por_participante
        .pivot(
            index="participante_id",
            columns="sistema_tts",
            values=metrica
        )
        .reindex(columns=ORDEN_SISTEMAS)
    )

    if tabla.isna().any().any():
        raise ValueError(
            f"Hay datos faltantes en tabla Friedman: {metrica}"
        )

    estadistico, p_value = friedmanchisquare(
        *[
            tabla[sistema].values
            for sistema in ORDEN_SISTEMAS
        ]
    )

    n = len(tabla)
    k = len(ORDEN_SISTEMAS)

    kendall_w = (
        estadistico /
        (n * (k - 1))
    )

    resultados_friedman.append({
        "metrica": metrica.replace("_media", ""),
        "n_participantes": n,
        "n_sistemas": k,
        "friedman_chi2": estadistico,
        "p_value": p_value,
        "kendall_W": kendall_w,
    })

friedman_df = pd.DataFrame(
    resultados_friedman
)

friedman_df[
    [
        "friedman_chi2",
        "p_value",
        "kendall_W",
    ]
] = (
    friedman_df[
        [
            "friedman_chi2",
            "p_value",
            "kendall_W",
        ]
    ]
    .round(8)
)

friedman_df.to_csv(
    TABLAS /
    "friedman_perceptual.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. WILCOXON PAREADO + HOLM
# ============================================================

resultados_wilcoxon = []

pares = list(
    combinations(
        ORDEN_SISTEMAS,
        2
    )
)

for metrica in [
    "similitud_media",
    "naturalidad_media",
]:

    tabla = (
        por_participante
        .pivot(
            index="participante_id",
            columns="sistema_tts",
            values=metrica
        )
        .reindex(columns=ORDEN_SISTEMAS)
    )

    resultados_metrica = []

    for sistema_1, sistema_2 in pares:

        x = tabla[sistema_1].values
        y = tabla[sistema_2].values

        diferencia = x - y

        # Si todas las diferencias fueran exactamente cero,
        # Wilcoxon no es aplicable.
        if np.allclose(diferencia, 0):

            estadistico = 0.0
            p_value = 1.0

        else:

            resultado = wilcoxon(
                x,
                y,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto"
            )

            estadistico = resultado.statistic
            p_value = resultado.pvalue

        resultados_metrica.append({
            "metrica": metrica.replace("_media", ""),
            "sistema_1": sistema_1,
            "sistema_2": sistema_2,
            "media_sistema_1": np.mean(x),
            "media_sistema_2": np.mean(y),
            "diferencia_media": np.mean(diferencia),
            "mediana_diferencia": np.median(diferencia),
            "wilcoxon_W": estadistico,
            "p_sin_corregir": p_value,
        })

    p_originales = [
        r["p_sin_corregir"]
        for r in resultados_metrica
    ]

    p_ajustados = correccion_holm(
        p_originales
    )

    for resultado, p_holm in zip(
        resultados_metrica,
        p_ajustados
    ):

        resultado["p_holm"] = p_holm
        resultado["significativo_0_05"] = (
            p_holm < 0.05
        )

        resultados_wilcoxon.append(
            resultado
        )

wilcoxon_df = pd.DataFrame(
    resultados_wilcoxon
)

columnas_redondear = [
    "media_sistema_1",
    "media_sistema_2",
    "diferencia_media",
    "mediana_diferencia",
    "wilcoxon_W",
    "p_sin_corregir",
    "p_holm",
]

wilcoxon_df[columnas_redondear] = (
    wilcoxon_df[columnas_redondear]
    .round(8)
)

wilcoxon_df.to_csv(
    TABLAS /
    "wilcoxon_holm_perceptual.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. GRAFICAS PERCEPTUALES
# ============================================================

guardar_boxplot(
    evaluaciones,
    "similitud",
    "Similitud perceptual por sistema",
    "Puntuación de similitud (1–5)",
    "boxplot_similitud_perceptual.png"
)

guardar_boxplot(
    evaluaciones,
    "naturalidad",
    "Naturalidad perceptual por sistema",
    "Puntuación de naturalidad (1–5)",
    "boxplot_naturalidad_perceptual.png"
)

guardar_barras_medias(
    resumen_participantes,
    "similitud_media",
    "similitud_ci95",
    "Similitud perceptual media por sistema",
    "Puntuación media (1–5)",
    "barras_similitud_perceptual_ic95.png"
)

guardar_barras_medias(
    resumen_participantes,
    "naturalidad_media",
    "naturalidad_ci95",
    "Naturalidad perceptual media por sistema",
    "Puntuación media (1–5)",
    "barras_naturalidad_perceptual_ic95.png"
)

guardar_distribucion(
    distribucion,
    "similitud",
    "Distribución de puntuaciones de similitud",
    "distribucion_similitud_perceptual.png"
)

guardar_distribucion(
    distribucion,
    "naturalidad",
    "Distribución de puntuaciones de naturalidad",
    "distribucion_naturalidad_perceptual.png"
)

guardar_por_hablante(
    por_hablante,
    "similitud_media",
    "Similitud perceptual por hablante y sistema",
    "Puntuación media de similitud",
    "similitud_por_hablante_sistema.png"
)

guardar_por_hablante(
    por_hablante,
    "naturalidad_media",
    "Naturalidad perceptual por hablante y sistema",
    "Puntuación media de naturalidad",
    "naturalidad_por_hablante_sistema.png"
)


# ============================================================
# 9. COMPARACION CON METRICAS OBJETIVAS DEL ENTREGABLE 3
# ============================================================

if ARCHIVO_OBJETIVAS.exists():

    objetivas = pd.read_csv(
        ARCHIVO_OBJETIVAS,
        encoding="utf-8-sig"
    )

    objetivas["sistema_tts"] = (
        objetivas["sistema_tts"]
        .apply(normalizar_sistema)
    )

    perceptual_para_merge = resumen_global[
        [
            "sistema_tts",
            "similitud_media",
            "naturalidad_media",
        ]
    ].copy()

    comparacion = pd.merge(
        perceptual_para_merge,
        objetivas,
        on="sistema_tts",
        how="inner"
    )

    comparacion = (
        comparacion
        .set_index("sistema_tts")
        .reindex(ORDEN_SISTEMAS)
        .reset_index()
    )

    comparacion.to_csv(
        TABLAS /
        "comparacion_objetiva_perceptual.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Correlaciones exploratorias
    #
    # IMPORTANTE:
    # Solo existen cuatro sistemas.
    # Estas correlaciones deben interpretarse descriptivamente,
    # no como evidencia inferencial fuerte.
    # --------------------------------------------------------

    pares_correlacion = [
        (
            "similitud_media",
            "ecapa_sim_media",
            "Similitud perceptual vs ECAPA"
        ),
        (
            "similitud_media",
            "eer_porcentaje",
            "Similitud perceptual vs EER"
        ),
        (
            "similitud_media",
            "auc_roc",
            "Similitud perceptual vs AUC ROC"
        ),
        (
            "similitud_media",
            "mfcc_dtw_media",
            "Similitud perceptual vs MFCC-DTW"
        ),
        (
            "naturalidad_media",
            "nisqa_mos_media",
            "Naturalidad perceptual vs NISQA-TTS"
        ),
    ]

    correlaciones = []

    for perceptual_col, objetiva_col, descripcion in pares_correlacion:

        if objetiva_col not in comparacion.columns:
            continue

        rho, p_value = spearmanr(
            comparacion[perceptual_col],
            comparacion[objetiva_col]
        )

        correlaciones.append({
            "comparacion": descripcion,
            "metrica_perceptual": perceptual_col,
            "metrica_objetiva": objetiva_col,
            "n_sistemas": len(comparacion),
            "spearman_rho": rho,
            "p_value": p_value,
            "nota": (
                "Interpretacion exploratoria: solo 4 sistemas."
            ),
        })

    correlaciones_df = pd.DataFrame(
        correlaciones
    )

    if not correlaciones_df.empty:

        correlaciones_df[
            ["spearman_rho", "p_value"]
        ] = (
            correlaciones_df[
                ["spearman_rho", "p_value"]
            ]
            .round(8)
        )

        correlaciones_df.to_csv(
            TABLAS /
            "correlaciones_objetivo_perceptual.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # --------------------------------------------------------
    # Graficas objetivo vs percepcion
    # --------------------------------------------------------

    if "ecapa_sim_media" in comparacion.columns:

        guardar_scatter(
            comparacion,
            "ecapa_sim_media",
            "similitud_media",
            "Similitud ECAPA-TDNN",
            "Similitud perceptual media (1–5)",
            "Similitud perceptual vs ECAPA-TDNN",
            "scatter_perceptual_vs_ecapa.png"
        )

    if "nisqa_mos_media" in comparacion.columns:

        guardar_scatter(
            comparacion,
            "nisqa_mos_media",
            "naturalidad_media",
            "NISQA-TTS MOS estimado",
            "Naturalidad perceptual media (1–5)",
            "Naturalidad perceptual vs NISQA-TTS",
            "scatter_perceptual_vs_nisqa.png"
        )

else:

    print()
    print(
        "ADVERTENCIA: no se encontro el resumen general "
        "del Entregable 3."
    )


# ============================================================
# 10. MOSTRAR RESULTADOS PRINCIPALES
# ============================================================

print()
print("============================================================")
print(" RESUMEN GLOBAL")
print("============================================================")
print()

print(
    resumen_global.to_string(
        index=False
    )
)

print()
print("============================================================")
print(" FRIEDMAN + KENDALL W")
print("============================================================")
print()

print(
    friedman_df.to_string(
        index=False
    )
)

print()
print("============================================================")
print(" WILCOXON + HOLM")
print("============================================================")
print()

print(
    wilcoxon_df[
        [
            "metrica",
            "sistema_1",
            "sistema_2",
            "diferencia_media",
            "p_sin_corregir",
            "p_holm",
            "significativo_0_05",
        ]
    ].to_string(
        index=False
    )
)

print()
print("============================================================")
print(" ARCHIVOS GENERADOS")
print("============================================================")
print()

print("Tablas:")
for archivo in sorted(TABLAS.glob("*.csv")):
    print(" -", archivo.name)

print()
print("Graficas:")
for archivo in sorted(GRAFICAS.glob("*.png")):
    print(" -", archivo.name)

print()
print("Analisis completado correctamente.")
print()
