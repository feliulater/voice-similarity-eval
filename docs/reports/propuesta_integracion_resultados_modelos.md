# Propuesta de integración de resultados, modelos y condiciones experimentales

## 1. Objetivo

Esta propuesta tiene como objetivo plantear una reorganización del repositorio `voice-similarity-eval` para que pueda utilizarse de forma más ordenada en el proyecto eléctrico. La estructura actual permite ejecutar un caso base, pero depende fuertemente de una carpeta plana de audios (`data/raw/`) y de convenciones de nombres para inferir identidad, tipo de audio y sistema utilizado.

Para etapas posteriores del proyecto, se requiere una organización que permita comparar voces naturales y sintéticas bajo diferentes condiciones, por ejemplo:

- voces infantiles,
- voces adultas,
- grupos etarios específicos,
- diferentes sistemas TTS,
- diferentes frases,
- diferentes condiciones de grabación,
- diferentes métricas de evaluación.

La propuesta busca facilitar la trazabilidad de los audios, modelos, resultados y configuraciones utilizadas en cada ejecución.

---

## 2. Limitación de la estructura actual

La estructura actual del repositorio funciona para una prueba inicial, pero presenta limitaciones para un proyecto más amplio:

```text
data/
  raw/

outputs/
  graficas/

scripts/
  evaluacion/
  generacion/
```

Las principales limitaciones identificadas son:

1. Todos los audios se encuentran en una sola carpeta.
2. La identidad del hablante se infiere desde el nombre del archivo.
3. El tipo de audio, real o sintético, se identifica mediante palabras clave como `sintetico` o `xtts`.
4. No existe un archivo central de metadatos.
5. Los resultados no siempre se guardan en archivos estructurados.
6. No hay separación clara por sistema TTS, grupo de hablantes, condición experimental o métrica.
7. Las dependencias externas, como NISQA, no están completamente integradas al flujo del repositorio.

---

## 3. Estructura propuesta del repositorio

Se propone reorganizar el repositorio de forma que permita separar datos, modelos, configuraciones, resultados y documentación. Una posible estructura sería:

```text
voice-similarity-eval/
│
├── data/
│   ├── natural/
│   │   ├── adultos/
│   │   ├── infantiles/
│   │   └── otros_grupos/
│   │
│   ├── synthetic/
│   │   ├── xtts/
│   │   ├── elevenlabs/
│   │   ├── openvoice/
│   │   └── otros_sistemas/
│   │
│   ├── prompts/
│   │   └── prompts.csv
│   │
│   └── metadata.csv
│
├── models/
│   ├── speaker_verification/
│   ├── quality_assessment/
│   └── tts/
│
├── results/
│   ├── runs/
│   │   └── YYYYMMDD_nombre_experimento/
│   │       ├── identidad/
│   │       ├── frecuencia/
│   │       ├── calidad/
│   │       ├── perceptual/
│   │       └── resumen/
│   │
│   └── comparaciones/
│
├── scripts/
│   ├── evaluacion/
│   ├── generacion/
│   ├── preparacion_datos/
│   └── utilidades/
│
├── docs/
│   ├── reports/
│   └── metodologia/
│
├── configs/
│   └── experimento_base.yaml
│
├── README.md
├── requirements.txt
└── Makefile
```

---

## 4. Organización de los datos

### 4.1 Voces naturales

Las voces naturales deberían organizarse por grupo o condición experimental. Por ejemplo:

```text
data/natural/adultos/
data/natural/infantiles/
data/natural/adultos_mayores/
```

Dentro de cada grupo, los archivos podrían organizarse por identificadores anónimos de hablante:

```text
data/natural/adultos/spk001/spk001_real_prompt001.wav
data/natural/adultos/spk001/spk001_real_prompt002.wav
data/natural/adultos/spk002/spk002_real_prompt001.wav
```

Esto evita depender de nombres personales y permite trabajar con identificadores consistentes.

### 4.2 Voces sintéticas

Las voces sintéticas deberían organizarse por sistema TTS y por hablante:

```text
data/synthetic/xtts/spk001/spk001_xtts_prompt001.wav
data/synthetic/elevenlabs/spk001/spk001_elevenlabs_prompt001.wav
data/synthetic/openvoice/spk001/spk001_openvoice_prompt001.wav
```

Esta estructura permite comparar diferentes sistemas TTS para una misma voz, frase y condición experimental.

---

## 5. Archivo de metadatos

Se recomienda crear un archivo `metadata.csv` como fuente principal de información del dataset. Este archivo evitaría que los scripts dependan únicamente del nombre de los archivos. El uso de metadatos permitiría filtrar los audios por grupo, condición, sistema TTS, frase o hablante sin modificar manualmente los scripts.

Una propuesta inicial de columnas es:

```text
audio_id,
speaker_id,
grupo,
edad_grupo,
tipo_audio,
sistema_tts,
modelo_version,
prompt_id,
texto,
condicion_grabacion,
sample_rate,
duracion_segundos,
ruta_archivo
```



---

## 6. Organización de modelos

Se recomienda separar los modelos utilizados para evaluación y generación:

```text
models/
  speaker_verification/
    ecapa_tdnn_speechbrain.txt

  quality_assessment/
    nisqa.txt

  tts/
    xtts_v2.txt
    elevenlabs.txt
    openvoice.txt
```

Estos archivos no necesariamente deben contener los pesos de los modelos, especialmente si son pesados o externos. Pueden funcionar como archivos de documentación con:

- nombre del modelo,
- fuente,
- versión,
- propósito,
- comando o procedimiento de instalación,
- fecha de uso,
- observaciones de compatibilidad.

Por ejemplo, para NISQA se podría documentar:

```text
Modelo: NISQA
Uso: estimación de calidad perceptual / MOS
Fuente: https://github.com/gabrielmittag/NISQA
Observación: debe clonarse manualmente en la raíz del proyecto para que funcione el script actual.
```

---

## 7. Organización de resultados

Se recomienda guardar los resultados por ejecución experimental. Esto permite comparar diferentes corridas sin sobrescribir archivos.

Ejemplo:

```text
results/runs/20260817_caso_base/
  identidad/
    eer_resultados.csv
    curva_roc_final.png
    matriz_similitud.png

  frecuencia/
    mcd_resultados.csv
    espectrograma_daniel.png
    espectrograma_felipe.png

  calidad/
    NISQA_results.csv
    resumen_mos.csv

  resumen/
    resumen_general.csv
    notas_ejecucion.md
```

Cada corrida debería tener un archivo de notas con información como:

- fecha,
- comando ejecutado,
- dataset utilizado,
- cantidad de audios,
- sistemas TTS evaluados,
- métricas utilizadas,
- errores encontrados,
- dependencias especiales.

---

## 8. Configuración de experimentos

Para evitar modificar scripts manualmente, se propone utilizar archivos de configuración en la carpeta `configs/`.

Ejemplo:

```text
configs/experimento_base.yaml
configs/comparacion_xtts_elevenlabs.yaml
configs/voces_infantiles.yaml
```

Un archivo de configuración podría contener:

```yaml
nombre_experimento: caso_base_xtts
dataset_metadata: data/metadata.csv
grupo: adultos
sistemas_tts:
  - xtts
metricas:
  - identidad
  - frecuencia
  - calidad
sample_rate: 16000
output_dir: results/runs/20260817_caso_base
```

Esto haría más sencillo repetir experimentos y cambiar condiciones sin alterar el código fuente.

---

## 9. Cambios sugeridos para el pipeline

A partir de esta propuesta, los scripts actuales podrían adaptarse de la siguiente forma:

1. Leer los audios desde `metadata.csv`.
2. Filtrar por grupo, sistema TTS, frase o condición experimental.
3. Guardar resultados de cada métrica en archivos CSV.
4. Guardar figuras en una carpeta de resultados asociada a cada ejecución.
5. Crear un resumen general por corrida.
6. Documentar dependencias externas como NISQA.
7. Evitar que la clasificación de audios dependa únicamente del nombre del archivo.

---

## 10. Cambios sugeridos para un futuro merge

Cuando los cambios desarrollados en el proyecto eléctrico se integren al repositorio principal, se recomienda incluir:

### Cambios incorporados

- Nueva estructura de carpetas para datos naturales y sintéticos.
- Archivo `metadata.csv` como fuente central de información.
- Exportación de resultados a CSV.
- Organización de resultados por corrida experimental.
- Documentación de modelos y dependencias externas.
- Actualización del README con instrucciones de ejecución.

### Cambios pendientes

- Integración completa de la generación de voces sintéticas.
- Automatización de la instalación de NISQA.
- Definición final del protocolo de evaluación perceptual.
- Implementación de pruebas ABX o MOS humanas.
- Comparación formal entre varios sistemas TTS.
- Incorporación de métricas dinámicas como ritmo o duración de segmentos.

---

## 11. Propuesta de README actualizado

El README principal debería incluir una sección breve con:

```text
1. Descripción del proyecto
2. Estructura del repositorio
3. Instalación del ambiente
4. Preparación del dataset
5. Ejecución de métricas
6. Ubicación de resultados
7. Dependencias externas
8. Problemas conocidos
9. Próximos pasos
```

Esto facilitaría que otra persona pueda reproducir el trabajo, depurar el repositorio o colaborar en módulos específicos.

---

## 12. Prioridad de implementación

Para no intentar modificar todo el repositorio de una sola vez, se propone dividir la reorganización en etapas.

### Etapa 1: Cambios mínimos necesarios

- Crear `metadata.csv`.
- Adaptar los scripts de evaluación para leer rutas y etiquetas desde `metadata.csv`.
- Guardar resultados principales en archivos CSV.
- Organizar los resultados por corrida experimental dentro de `results/runs/`.
- Documentar dependencias externas como NISQA.

### Etapa 2: Comparación entre sistemas TTS

- Separar voces sintéticas por sistema TTS.
- Agregar filtros por sistema TTS en los scripts.
- Generar resúmenes comparativos por sistema.
- Crear una configuración por experimento en `configs/`.

### Etapa 3: Evaluación perceptual y mejoras futuras

- Incorporar pruebas perceptuales humanas.
- Diseñar pruebas ABX o MOS.
- Evaluar atributos como acento, ritmo, naturalidad y prosodia.
- Explorar métricas dinámicas adicionales.

---

## 13. Conclusión

La estructura actual del repositorio permitió ejecutar un caso base y validar las métricas principales. Sin embargo, para el proyecto completo se necesita una organización más escalable que permita trabajar con diferentes grupos de hablantes, condiciones experimentales, sistemas TTS y métricas.

La propuesta presentada busca separar claramente los datos naturales, las voces sintéticas, los modelos, las configuraciones experimentales y los resultados. Además, se recomienda utilizar metadatos estructurados y guardar resultados en archivos CSV para facilitar el análisis posterior.

Esta reorganización permitiría que el repositorio sea más reproducible, mantenible y útil para futuras comparaciones de sistemas de texto a voz en español costarricense.
