# Proyecto Eléctrico — Evaluación comparativa de sistemas TTS

Este directorio contiene los scripts desarrollados y utilizados por **Felipe Ulate Rodríguez** durante el Proyecto Eléctrico de la Escuela de Ingeniería Eléctrica de la Universidad de Costa Rica.

El trabajo estudia la generación y evaluación de voces sintéticas en español costarricense mediante cuatro sistemas de texto a voz y clonación/adaptación de voz:

- XTTS
- F5-TTS
- OpenVoice
- ElevenLabs

El flujo implementado abarca desde la generación de las voces sintéticas hasta su evaluación objetiva y perceptual.

Los archivos desarrollados específicamente para este proyecto se mantienen dentro de:

```text
scripts/proyecto_felipe/
```

Esto permite separar este trabajo del código previamente existente en el repositorio base.

---

## 1. Flujo general del proyecto

El trabajo se organizó en tres etapas principales:

```text
Entregable 2
Generación de voces sintéticas
        ↓
Entregable 3
Evaluación objetiva
        ↓
Entregable 4
Evaluación perceptual e integración final
```

### Entregable 2

Generación de las voces sintéticas mediante:

- XTTS
- F5-TTS
- OpenVoice
- ElevenLabs

### Entregable 3

Evaluación objetiva mediante:

- similitud de hablante con ECAPA-TDNN;
- EER y curvas ROC;
- distancia cepstral MFCC-DTW;
- naturalidad estimada mediante NISQA-TTS;
- control de calidad de los audios.

### Entregable 4

Evaluación perceptual mediante participantes humanos y análisis estadístico de:

- similitud con la voz natural de referencia;
- naturalidad percibida.

También se realiza la comparación descriptiva entre los resultados perceptuales y las métricas objetivas obtenidas en el Entregable 3.

---

# 2. Estructura del código

```text
proyecto_felipe/
├── entregable_2_generacion/
│   ├── xtts/
│   │   ├── generar_xtts_todos_raw.py
│   │   ├── postprocesar_xtts.py
│   │   └── actualizar_metadata_xtts.py
│   │
│   ├── f5tts/
│   │   ├── generar_f5tts_todos.py
│   │   └── actualizar_metadata_f5tts.py
│   │
│   ├── openvoice/
│   │   ├── generar_openvoice_todos.py
│   │   └── actualizar_metadata_openvoice.py
│   │
│   ├── elevenlabs/
│   │   └── segmentar_elevenlabs_v2.py
│   │
│   └── utilidades/
│       └── verificar_estado_entregable2.py
│
├── entregable_3_metricas/
│   ├── calcular_distancia_mfcc_dtw.py
│   ├── calcular_eer_roc_ecapa.py
│   ├── calcular_similitud_ecapa.py
│   ├── consolidar_metricas_entregable3.py
│   ├── control_calidad_audios.py
│   ├── graficar_distancia_mfcc_dtw.py
│   ├── graficar_distribuciones_verificacion.py
│   ├── graficar_nisqa_tts.py
│   ├── graficar_similitud_ecapa.py
│   └── preparar_nisqa_tts.py
│
├── entregable_4_perceptual/
│   ├── analizar_evaluacion_perceptual.py
│   ├── consolidar_resultados_finales.py
│   ├── preparar_evaluacion_perceptual.py
│   └── procesar_evaluacion_perceptual.py
│
├── README.md
└── requirements_proyecto_felipe.txt
```

---

# 3. Entregable 2 — Generación de voces sintéticas

El segundo entregable corresponde a la generación del conjunto de voces sintéticas utilizado posteriormente en las evaluaciones.

El conjunto final del proyecto contiene cinco hablantes identificados internamente como:

```text
spk001
spk002
spk003
spk004
spk005
```

Se utilizaron 25 prompts por hablante.

Por cada sistema se generaron 125 audios sintéticos, para un total de:

```text
125 voces naturales
125 XTTS
125 F5-TTS
125 OpenVoice
125 ElevenLabs

625 audios totales
```

---

## 3.1 XTTS

Se utilizó **XTTS v2** mediante la librería Coqui TTS.

### Scripts

#### `generar_xtts_todos_raw.py`

Genera inicialmente las voces sintéticas utilizando XTTS v2.

La selección del dispositivo se realiza automáticamente:

```text
CUDA si existe una GPU disponible
CPU en caso contrario
```

Las salidas iniciales se almacenan en una carpeta intermedia correspondiente a `xtts_raw`.

#### `postprocesar_xtts.py`

Realiza el postprocesamiento de las salidas generadas con XTTS.

El procedimiento incluye tratamiento de regiones activas de la señal y preparación de los audios finales utilizados en las evaluaciones.

#### `actualizar_metadata_xtts.py`

Actualiza los archivos de metadatos y registra el estado final de los audios generados.

---

## 3.2 F5-TTS

F5-TTS fue instalado y utilizado dentro de un entorno virtual de Python en Ubuntu.

La generación se realizó mediante la interfaz de línea de comandos:

```bash
f5-tts_infer-cli
```

Durante el desarrollo original se utilizó una instalación controlada mediante:

```bash
pip install f5-tts --no-deps
```

debido a las limitaciones de almacenamiento disponibles en la partición Linux.

En una instalación sin esta limitación puede utilizarse la instalación estándar correspondiente a F5-TTS.

### Scripts

#### `generar_f5tts_todos.py`

Automatiza la generación del conjunto completo mediante F5-TTS.

El script utiliza:

- audio de referencia;
- texto correspondiente al audio de referencia;
- texto objetivo;
- F5-TTS;
- GPU cuando está disponible.

Las cachés de modelos pueden configurarse mediante una variable de entorno para evitar almacenarlas en la partición principal.

#### `actualizar_metadata_f5tts.py`

Actualiza los metadatos y estados asociados con las voces generadas mediante F5-TTS.

---

# 3.3 OpenVoice + MeloTTS

La implementación utilizada combina dos herramientas:

- MeloTTS
- OpenVoice V2

El flujo utilizado fue:

```text
Texto
  ↓
MeloTTS
  ↓
Voz base en español
  ↓
OpenVoice V2
  ↓
Transferencia de características del hablante de referencia
  ↓
Voz sintética final
```

MeloTTS genera primero una voz base en español.

Posteriormente OpenVoice utiliza esa señal y le transfiere características de la voz del hablante de referencia.

### Repositorios externos

OpenVoice y MeloTTS se utilizaron a partir de sus respectivos repositorios de código y no se incluyen como copia dentro de este directorio.

### Scripts

#### `generar_openvoice_todos.py`

Automatiza el proceso completo de generación:

1. carga los prompts;
2. genera una voz base con MeloTTS;
3. obtiene las características del hablante de referencia;
4. aplica OpenVoice V2;
5. almacena la salida final.

#### `actualizar_metadata_openvoice.py`

Actualiza los metadatos correspondientes a las voces generadas con OpenVoice.

---

# 3.4 ElevenLabs

ElevenLabs se utilizó como sistema comercial de referencia.

La generación original se realizó utilizando el servicio de ElevenLabs y no está automatizada dentro de este repositorio.

Las salidas obtenidas originalmente correspondían a archivos de mayor duración que contenían múltiples frases.

### Script

#### `segmentar_elevenlabs_v2.py`

Realiza:

- detección de pausas;
- segmentación de las grabaciones;
- extracción de los prompts individuales;
- conversión a WAV;
- organización de los archivos finales.

La versión `v2` corresponde al script final utilizado en el proyecto.

---

# 3.5 Verificación del Entregable 2

### `verificar_estado_entregable2.py`

Realiza comprobaciones sobre:

- número de audios naturales;
- número de audios generados por sistema;
- distribución por hablante;
- existencia de metadatos;
- organización general del conjunto.

---

# 4. Entregable 3 — Evaluación objetiva

El tercer entregable corresponde a la evaluación automática de las voces sintéticas.

Se utilizaron cuatro grupos principales de análisis:

1. similitud de hablante;
2. verificación de hablante;
3. semejanza espectral;
4. naturalidad estimada.

También se realizó control general de calidad de los audios.

---

## 4.1 ECAPA-TDNN

Se utilizaron embeddings de hablante mediante el modelo:

```text
speechbrain/spkrec-ecapa-voxceleb
```

y la implementación de SpeechBrain.

### `calcular_similitud_ecapa.py`

Calcula embeddings para las voces naturales y sintéticas.

Posteriormente calcula similitud coseno entre cada voz sintética y su referencia natural correspondiente.

Todos los audios se convierten internamente a:

```text
16 kHz
mono
```

cuando es necesario.

El remuestreo se realiza en memoria y no modifica los archivos originales.

---

# 4.2 Verificación de hablante — EER y ROC

### `calcular_eer_roc_ecapa.py`

Construye pares positivos y negativos de hablantes utilizando embeddings ECAPA-TDNN.

Para cada muestra sintética se utilizaron:

```text
1 par positivo
4 pares negativos
```

Se calcularon:

- Equal Error Rate (EER);
- umbral asociado al EER;
- curva ROC;
- área bajo la curva ROC.

---

# 4.3 MFCC-DTW

### `calcular_distancia_mfcc_dtw.py`

Calcula una medida de semejanza espectral utilizando:

- MFCC;
- Dynamic Time Warping.

Configuración utilizada:

```text
Frecuencia de muestreo: 16 kHz
MFCC: 13 coeficientes calculados
Coeficientes utilizados: MFCC 1-12
n_fft: 1024
hop_length: 256
Mel bands: 40
fmin: 20 Hz
fmax: 7600 Hz
```

Los silencios iniciales y finales se recortan utilizando un criterio de:

```text
35 dB
```

Posteriormente se aplica DTW y se calcula la distancia euclidiana media a lo largo del camino de alineación.

La métrica se reporta como:

```text
distancia cepstral MFCC-DTW
```

Una distancia menor representa mayor semejanza espectral.

---

# 4.4 NISQA-TTS

NISQA-TTS se utilizó como estimador automático de naturalidad.

### `preparar_nisqa_tts.py`

Prepara la entrada requerida por NISQA a partir del manifiesto general de evaluación.

### `graficar_nisqa_tts.py`

Procesa los resultados obtenidos y genera tablas y visualizaciones.

NISQA se utiliza como dependencia externa y no se incluye como copia dentro de `scripts/proyecto_felipe`.

La puntuación estimada se interpreta de forma aproximada como una medida de naturalidad/calidad percibida.

Una puntuación mayor representa una mejor estimación de naturalidad.

NISQA no sustituye la evaluación perceptual humana.

---

# 4.5 Control de calidad de audios

### `control_calidad_audios.py`

Analiza los 625 audios del conjunto.

Se calculan características como:

- duración total;
- duración después del recorte de silencios;
- silencio en los extremos;
- amplitud pico;
- RMS;
- clipping.

El propósito de este análisis es detectar diferencias técnicas importantes entre las fuentes antes de interpretar las métricas de similitud o naturalidad.

---

# 4.6 Gráficas y consolidación

Los siguientes scripts generan las principales visualizaciones:

- `graficar_similitud_ecapa.py`
- `graficar_distribuciones_verificacion.py`
- `graficar_distancia_mfcc_dtw.py`
- `graficar_nisqa_tts.py`

### `consolidar_metricas_entregable3.py`

Integra los resultados de:

- ECAPA-TDNN;
- EER;
- AUC ROC;
- MFCC-DTW;
- NISQA-TTS;
- control de calidad.

Las métricas no deben interpretarse como una única puntuación global, ya que representan dimensiones diferentes del desempeño de los sistemas.

---

# 5. Entregable 4 — Evaluación perceptual

El cuarto entregable incorpora una evaluación humana de las voces generadas.

La evaluación se realizó con:

```text
20 participantes
```

distribuidos entre:

```text
4 versiones del formulario
5 participantes por versión
```

Cada participante evaluó voces de los cuatro sistemas.

---

# 5.1 Diseño perceptual

Se evaluaron dos dimensiones.

## Similitud

Pregunta relacionada con qué tan similar se percibe la voz sintética respecto a la voz natural del hablante.

Escala:

```text
1 = Nada similar
2 = Poco similar
3 = Moderadamente similar
4 = Bastante similar
5 = Muy similar
```

## Naturalidad

Evalúa qué tan natural o artificial se percibe cada voz.

Escala:

```text
1 = Muy artificial
2 = Artificial
3 = Intermedia
4 = Natural
5 = Muy natural
```

---

# 5.2 Preparación de la evaluación

### `preparar_evaluacion_perceptual.py`

Genera una asignación balanceada de los estímulos.

Se utilizaron cuatro versiones:

```text
A
B
C
D
```

Las etiquetas visibles para los participantes fueron únicamente:

```text
Audio A
Audio B
Audio C
Audio D
```

La identidad real de los sistemas permaneció oculta durante la evaluación.

El diseño se balanceó para que cada sistema apareciera de forma equivalente bajo las diferentes etiquetas ciegas.

---

# 5.3 Procesamiento de respuestas

### `procesar_evaluacion_perceptual.py`

Lee las respuestas exportadas desde Google Forms y reconstruye la identidad real de cada sistema utilizando la asignación interna.

El resultado consolidado contiene:

```text
400 evaluaciones de audio
100 evaluaciones por sistema
80 evaluaciones por hablante
```

Cada evaluación contiene una puntuación de:

- similitud;
- naturalidad.

---

# 5.4 Análisis estadístico

### `analizar_evaluacion_perceptual.py`

Realiza el análisis estadístico de las respuestas.

Para evitar considerar las múltiples evaluaciones de un mismo participante como observaciones completamente independientes, se calculan medias por participante y sistema.

Se aplican:

- estadística descriptiva;
- prueba de Friedman;
- coeficiente de concordancia de Kendall;
- prueba pareada de Wilcoxon;
- corrección de Holm para comparaciones múltiples.

También se generan:

- gráficas de medias;
- intervalos de confianza;
- distribuciones;
- boxplots;
- análisis por hablante.

---

# 5.5 Integración objetiva + perceptual

### `consolidar_resultados_finales.py`

Integra las principales métricas del Entregable 3 con los resultados perceptuales del Entregable 4.

Se comparan descriptivamente:

```text
ECAPA-TDNN
EER
AUC ROC
MFCC-DTW
NISQA-TTS
Similitud perceptual
Naturalidad perceptual
```

El objetivo no es producir una puntuación global única, sino analizar coincidencias y discrepancias entre diferentes dimensiones de evaluación.

---

# 6. Configuración de rutas

Las versiones originales de los scripts se ejecutaron dentro de la estructura local utilizada durante el proyecto.

Antes de incorporarlos al repositorio, las rutas específicas de la computadora de desarrollo fueron reemplazadas por variables de entorno.

---

## 6.1 Ruta general del proyecto

Para los Entregables 3 y 4:

```bash
export PROYECTO_TTS_ROOT="/ruta/Proyecto_Electrico_Voces_TTS"
```

Si no se define, se utiliza de forma predeterminada:

```text
~/Proyecto_Electrico_Voces_TTS
```

---

## 6.2 Ruta del Entregable 2

```bash
export PROYECTO_TTS_E2_DIR="/ruta/entregable_2_voces"
```

---

## 6.3 Caché de modelos

```bash
export TTS_MODEL_CACHE="/ruta/cache_tts"
```

Si no se define, se utiliza una carpeta dentro de:

```text
~/.cache/
```

---

## 6.4 OpenVoice

```bash
export OPENVOICE_DIR="/ruta/OpenVoice"
```

---

## 6.5 MeloTTS

```bash
export MELOTTS_DIR="/ruta/MeloTTS"
```

---

# 7. Dependencias principales

Las principales bibliotecas de Python utilizadas durante el proyecto incluyen:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
librosa
soundfile
torch
torchaudio
speechbrain
TTS
unidic-lite
```

El archivo:

```text
requirements_proyecto_felipe.txt
```

contiene las dependencias generales del código incluido en este directorio.

---

# 8. Dependencias externas

Algunas herramientas requieren instalación independiente y no están contenidas dentro de este directorio.

## F5-TTS

Debe instalarse F5-TTS y estar disponible:

```bash
f5-tts_infer-cli
```

## OpenVoice

Debe existir una instalación local del repositorio OpenVoice.

La ruta puede configurarse mediante:

```bash
export OPENVOICE_DIR="/ruta/OpenVoice"
```

## MeloTTS

Debe existir una instalación local de MeloTTS.

```bash
export MELOTTS_DIR="/ruta/MeloTTS"
```

## NISQA

NISQA debe instalarse de forma independiente.

Durante el proyecto se utilizó el modelo:

```text
NISQA_TTS_v1
```

## FFmpeg

Algunos scripts del Entregable 2 utilizan:

```text
ffmpeg
ffprobe
```

por lo que ambos deben estar disponibles en el sistema.

---

# 9. Orden general de ejecución

El flujo completo puede resumirse de la siguiente forma.

## Etapa 1 — Generación

```text
Preparar audios naturales y prompts
        ↓
XTTS
F5-TTS
OpenVoice + MeloTTS
ElevenLabs
        ↓
Actualizar metadatos
        ↓
Verificar conjunto
```

## Etapa 2 — Evaluación objetiva

```text
Preparar manifest
        ↓
ECAPA-TDNN
        ↓
EER / ROC
        ↓
MFCC-DTW
        ↓
NISQA-TTS
        ↓
Control de calidad
        ↓
Gráficas
        ↓
Consolidación
```

## Etapa 3 — Evaluación perceptual

```text
Preparar asignaciones
        ↓
Crear versiones de encuesta
        ↓
Recolectar respuestas
        ↓
Procesar CSV
        ↓
Análisis estadístico
        ↓
Integración objetivo + perceptual
```

---

# 10. Datos no incluidos en el repositorio

Por razones de privacidad, tamaño de archivos y manejo ético del material, este repositorio no incluye:

- grabaciones naturales de los hablantes;
- voces sintéticas generadas;
- consentimientos firmados;
- información personal de los hablantes;
- respuestas crudas de la evaluación perceptual;
- pesos de modelos;
- cachés de modelos;
- entornos virtuales;
- archivos temporales.

Los scripts esperan que los datos requeridos se encuentren disponibles localmente cuando se desee reproducir un experimento.

---

# 11. Consideraciones de privacidad

Los identificadores utilizados para los hablantes dentro del proyecto tienen la forma:

```text
spk001
spk002
spk003
spk004
spk005
```

El repositorio no requiere publicar nombres reales ni otra información personal asociada con los participantes.

Las respuestas perceptuales originales tampoco forman parte del código publicado.

---

# 12. Notas de reproducibilidad

Las versiones incluidas en este directorio fueron adaptadas para eliminar rutas absolutas específicas de la computadora utilizada durante el desarrollo.

Antes de incorporarlas al repositorio se verificó:

- ausencia de rutas personales;
- ausencia de credenciales;
- ausencia de API keys;
- sintaxis válida mediante `compileall`;
- exclusión de cachés `__pycache__`.

Las modificaciones de portabilidad no cambian la metodología utilizada originalmente durante los experimentos.

---

# 13. Herramientas principales utilizadas

Durante el proyecto se utilizaron:

- Python
- Ubuntu
- CUDA
- PyTorch
- Coqui TTS / XTTS
- F5-TTS
- OpenVoice
- MeloTTS
- ElevenLabs
- SpeechBrain
- ECAPA-TDNN
- librosa
- NISQA-TTS
- SciPy
- scikit-learn
- pandas
- Matplotlib
- FFmpeg

---

# 14. Autor

**Felipe Ulate Rodríguez**
Escuela de Ingeniería Eléctrica
Universidad de Costa Rica
II Ciclo 2026
