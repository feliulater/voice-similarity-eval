import os
import glob
import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve

print("1. Cargando el modelo ECAPA-TDNN...")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

print("2. Procesando audios de la carpeta data/raw/ ...")
rutas_audios = glob.glob("data/raw/*.wav")
metadata = []
embeddings = []

# Extraer biometría y metadatos de todos los audios
for ruta in sorted(rutas_audios):
    nombre = os.path.basename(ruta).lower()
    
    # Lógica de identificación basada en el nombre del archivo
    partes = nombre.replace('.wav', '').split('_')
    identidad = partes[0] # Ej: 'daniel' o 'felipe'
    
    tipo = "sintetico" if ("sintetico" in nombre or "xtts" in nombre) else "real"
    
    metadata.append({'id': identidad, 'tipo': tipo, 'archivo': nombre})
    
    # Procesamiento de audio
    # Se usa librosa para evitar dependencia de TorchCodec/CUDA al cargar WAV
    audio, fs = librosa.load(ruta, sr=16000, mono=True)
    signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        
    with torch.no_grad():
        emb = classifier.encode_batch(signal).squeeze().cpu().numpy()
        embeddings.append(emb)

print("3. Calculando cruces y clasificando accesos...")
y_true = []
y_scores = []

# Comparar todos contra todos (sin repetir y sin compararse consigo mismo)
for i in range(len(metadata)):
    for j in range(i + 1, len(metadata)):
        file_i = metadata[i]
        file_j = metadata[j]
        
        # Calcular similitud vectorial
        similitud = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        
        # LÓGICA DEL PROTOCOLO (Clasificación automática)
        # Caso 1: Genuino (Misma persona, audios reales)
        if file_i['id'] == file_j['id'] and file_i['tipo'] == 'real' and file_j['tipo'] == 'real':
            y_true.append(1)
            y_scores.append(similitud)
            
        # Caso 2: Impostor Natural (Personas distintas, audios reales)
        elif file_i['id'] != file_j['id'] and file_i['tipo'] == 'real' and file_j['tipo'] == 'real':
            y_true.append(0)
            y_scores.append(similitud)
            
        # Caso 3: Ataque Sintético (Misma persona, pero uno es clonado)
        elif file_i['id'] == file_j['id'] and (file_i['tipo'] == 'sintetico' or file_j['tipo'] == 'sintetico'):
            y_true.append(0)  # Es un ataque, el sistema DEBE rechazarlo
            y_scores.append(similitud)

print(f"-> Se evaluaron {len(y_scores)} cruces válidos.")

print("\n4. Calculando Umbral Estadístico (EER)...")
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
fnr = 1 - tpr

eer_idx = np.nanargmin(np.absolute((fpr - fnr)))
eer_threshold = thresholds[eer_idx]
eer = fpr[eer_idx]

print(f"--- RESULTADOS FINALES ---")
print(f"Umbral Óptimo (Threshold): {eer_threshold:.4f}")
print(f"Equal Error Rate (EER): {eer*100:.2f}%")

print("\n5. Generando Curva ROC...")
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (EER = {eer*100:.2f}%)')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsa Aceptación (FPR) - Ataques exitosos')
plt.ylabel('Tasa de Verdadera Aceptación (TPR) - Genuinos aceptados')
plt.title('Curva ROC - Protocolo Dinámico de Similitud de Voz LAPA-CR')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.savefig("curva_roc_final.png", dpi=300)
print("¡Proceso completado! Gráfica guardada como 'curva_roc_final.png'.")
