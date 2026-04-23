from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import librosa
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
    
    os.remove(temp_path) # Limpiar
    
    return {
        "bpm": round(float(tempo), 2),
        "beats": [round(float(b), 3) for b in beat_times]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)