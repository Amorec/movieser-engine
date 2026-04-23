from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import os

app = FastAPI(title="MovieSer Music Engine")

# Esto permite que tu Next.js hable con este servidor
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "MovieSer Engine Online", "message": "Hola Claudia, el servidor esta activo"}

def detect_mood(y, sr, bpm):
    """Detecta el mood basado en BPM, energía y características espectrales."""
    # Energía RMS promedio
    rms = np.mean(librosa.feature.rms(y=y))
    # Centroide espectral promedio (brillo del sonido)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Clasificación por BPM + energía + brillo
    if bpm >= 120 and rms > 0.05 and spectral_centroid > 2000:
        return "energetic"     # Rápido, fuerte, brillante
    elif bpm >= 100 and rms > 0.03:
        return "happy"         # Moderado-rápido, con energía
    elif bpm <= 80 and rms < 0.03:
        return "sad"           # Lento, suave
    elif bpm <= 90 and spectral_centroid < 1500:
        return "calm"          # Lento, oscuro/suave
    elif bpm >= 110 and rms > 0.04 and spectral_centroid < 1800:
        return "dark"          # Rápido pero oscuro/pesado
    elif bpm <= 95 and rms < 0.04:
        return "romantic"      # Tempo medio-bajo, suave
    else:
        return "neutral"       # No encaja claramente


@app.post("/analyze-music")
async def analyze_music(file: UploadFile = File(...)):
    # Guardar archivo temporalmente
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Analizar ritmo con Librosa
    y, sr = librosa.load(temp_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    bpm = round(float(tempo), 2)

    # Detectar mood
    mood = detect_mood(y, sr, bpm)
    
    os.remove(temp_path) # Limpiar
    
    return {
        "bpm": bpm,
        "beats": [round(float(b), 3) for b in beat_times],
        "mood": mood
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)