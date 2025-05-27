"""
highlight_extractor.py
Detecta los 2-3 clips más interesantes de un vídeo (~3 min) usando:
- picos de audio (energía RMS por segundo, librosa)
- movimiento visual (diferencia de fotogramas, OpenCV)
Luego corta cada fragmento con FFmpeg.

Requisitos:
  pip install numpy librosa soundfile opencv-python moviepy
  ffmpeg.exe accesible en el PATH
"""

import os
import subprocess
import tempfile
import numpy as np
import librosa
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# ───────── CONFIGURACIÓN BÁSICA ───────── #
VIDEO_PATH   = "input.mp4"   # ruta al vídeo de prueba (~3 min)
TARGET_CLIPS = 1             # clips a extraer
WIN_SIZE     = 1.0           # ventana de análisis (s)
CLIP_LEN     = 30             # duración de cada clip (s)
AUDIO_W      = 2.0           # peso del audio en la puntuación
MOTION_W     = 1.0           # peso del movimiento
# ──────────────────────────────────────── #

def audio_peaks(video_path: str, win: float) -> np.ndarray:
    """Devuelve energía RMS por ventana."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = tmp.name
    # Extraer pista de audio mono 44.1 kHz
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "44100", wav],
        capture_output=True,
        check=True,
    )
    y, sr = librosa.load(wav, sr=None)
    hop = int(sr * win)
    rms = librosa.feature.rms(y=y, frame_length=hop, hop_length=hop, center=False)[0]
    os.remove(wav)
    return rms


def motion_peaks(video_path: str, win: float) -> np.ndarray:
    """Devuelve cantidad de movimiento medio por ventana."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps * win)
    prev = None
    mvals = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))  # baja resolución → más rápido
            if prev is not None:
                diff = cv2.absdiff(gray, prev)
                mvals.append(diff.sum())
            prev = gray
        idx += 1

    cap.release()
    return np.array(mvals)


def combined_score(audio: np.ndarray, motion: np.ndarray,
                   aw: float, mw: float) -> np.ndarray:
    """Normaliza y suma audio+vídeo según pesos."""
    a = (audio  - audio.mean())  / (audio.std()  + 1e-6)
    m = (motion - motion.mean()) / (motion.std() + 1e-6)
    n = min(len(a), len(m))
    return aw * a[:n] + mw * m[:n]


def best_segments(score: np.ndarray, win: float,
                  nclips: int, clip_len: int):
    """Devuelve [(start, end), …] en segundos, sin solaparse."""
    smooth = np.convolve(score, np.ones(3) / 3, mode="same")
    idx_desc = np.argsort(smooth)[::-1]          # índices ordenados desc.
    segs, taken = [], np.zeros_like(smooth, bool)
    half = int(clip_len / (2 * win))

    for i in idx_desc:
        if len(segs) >= nclips:
            break
        if taken[max(0, i - half): i + half + 1].any():
            continue
        start = max(0, (i - half) * win)
        end   = start + clip_len
        segs.append((start, end))
        taken[max(0, i - half): i + half + 1] = True
    return segs


def cut_clips(video_path: str, segments, outdir="clips"):
    os.makedirs(outdir, exist_ok=True)
    outputs = []
    for k, (s, e) in enumerate(segments, 1):
        name = os.path.join(outdir, f"clip{k}.mp4")
        # -ss después de -i y re-encode seguro
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(s),
            "-to", str(e),
            "-vf", "format=yuv420p",            # compatibilidad universal
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",          # que se reproduzca en streaming
            name
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        outputs.append(name)
    return outputs

def main():
    print("⏳ Analizando audio...")
    audio_rms = audio_peaks(VIDEO_PATH, WIN_SIZE)

    print("⏳ Analizando vídeo...")
    motion = motion_peaks(VIDEO_PATH, WIN_SIZE)

    print("⏳ Combinando puntuaciones...")
    score = combined_score(audio_rms, motion, AUDIO_W, MOTION_W)

    segments = best_segments(score, WIN_SIZE, TARGET_CLIPS, CLIP_LEN)
    print("🔖 Segmentos seleccionados:", segments)

    print("✂️  Cortando clips...")
    files = cut_clips(VIDEO_PATH, segments)
    print("✅ Clips generados:", files)


if __name__ == "__main__":
    main()
