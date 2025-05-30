"""
clipExtractor.py
Identifies and extracts potentially viral short clips from a longer video.

Core Functionality:
- Handles video input (local file or YouTube URL, with download capability).
- Transcribes video audio using Whisper (with word-level timestamps).
- Identifies candidate highlight segments by performing LLM analysis (e.g., Gemma-2B-IT)
  on the full transcript, using chunking and parallel processing for long transcripts.
- For each candidate, performs visual analysis (YOLO for object detection, FER for emotions).
- Calculates a 'virality score' for segments based on multi-modal analysis (text, visual, audio energy, motion).
- Selects top N non-overlapping segments based on this score.
- Cuts selected segments using FFmpeg.
- Logs performance metrics for major operations (transcription, LLM analysis, motion analysis FPS, VRAM usage).

Key Dependencies:
  numpy, librosa, opencv-python, moviepy, yt-dlp, validators, ffmpeg (system path),
  faster_whisper, transformers, torch, torchvision, yolov5, fer, bitsandbytes, accelerate.

Usage:
  Ensure all dependencies from requirements.txt are installed.
  Ensure 'ffmpeg' is accessible in the system PATH.
  The script is run asynchronously.
  Example: `python clipExtractor.py "your_video_file.mp4_or_youtube_url"`
"""

import torch
import os
import subprocess
import tempfile
import numpy as np
import librosa
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from typing import List, Dict, Optional
import json
from app.services.services.download_youtube_video import download_video
import validators
from transformers import BitsAndBytesConfig, pipeline
import torchvision.transforms.functional as TF
import logging
import argparse
from rich.console import Console
from rich.logging import RichHandler
import asyncio
import time

# --- Initialize Rich Console and Logging ---
console = Console()
LOG_LEVEL = logging.INFO # Default log level
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(message)s", # Rich handles formatting, so this is minimal
    datefmt="[%X]", # Timestamp format
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False, markup=True)] # Added markup=True
)
log = logging.getLogger("rich")
# --- End Logging Setup ---

# ───────── DEVICE CONFIGURATION ───────── #
FORCE_CPU = os.getenv("FORCE_CPU", "0").lower() in ["1", "true"]
if FORCE_CPU:
    DEVICE = torch.device("cpu")
    log.info("FORCE_CPU environment variable set. Using CPU.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    log.info("CUDA is available. Using GPU.")
else:
    DEVICE = torch.device("cpu")
    log.info("CUDA not available. Using CPU.")
# ────────────────────────────────────────── #

bnb_config_gpu = BitsAndBytesConfig( # Renamed for clarity, this is for GPU
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
# For CPU, 4-bit quantization with bitsandbytes is often not supported or optimal.
# bnb_config will be set dynamically in load_llm_model based on DEVICE.

# Attempt to import AI models; handle ImportError if they are not installed
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None # type: ignore
    log.warning("faster_whisper not installed. Transcription will be simulated if model cannot be loaded.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    AutoTokenizer = None # type: ignore
    AutoModelForCausalLM = None # type: ignore
    log.warning("transformers not installed. LLM-based text analysis will be simulated.")

try:
    # Assuming yolov5 is installed from a package that makes it importable
    class YOLOv5Placeholder: # Placeholder if actual import fails or is complex
        def __init__(self, weights, device): self.weights = weights; self.device = device; log.info(f"YOLOv5Placeholder initialized with weights: {weights} on device: {device}")
        def predict(self, frame): log.debug("YOLOv5Placeholder: predict called"); return [] 
    YOLOv5 = YOLOv5Placeholder # Default to placeholder

    try:
        from yolov5 import YOLOv5 as ActualYOLOv5 
        YOLOv5 = ActualYOLOv5
        log.info("Successfully imported 'yolov5.YOLOv5'.")
    except ImportError:
        try:
            import torch
            _yolo_model_hub_temp = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True) 
            log.info("Successfully checked torch.hub.load for ultralytics/yolov5.")
        except Exception as e_yolo_hub:
            log.warning(f"Could not import 'yolov5' or load from torch.hub ({e_yolo_hub}). YOLOv5 detection will be simulated by placeholder.")

except ImportError:
    log.warning("YOLOv5 related imports failed at placeholder definition. YOLOv5 detection will be simulated.")

try:
    from fer import FER
except ImportError:
    FER = None # type: ignore
    log.warning("fer (Facial Emotion Recognition) not installed. Emotion detection will be simulated.")


# Global variables for models (to be loaded once)
_stt_model = None
_llm = None
_tokenizer = None
_yolo_model = None
_fer_detector = None

# ───────── DIRECTORIES AND CONFIGS ───────── #
MODELS_DIR = "models" 
# For example: os.path.join(os.path.dirname(__file__), "models")

# ───────── CONFIGURACIÓN BÁSICA (Copied from previous state) ───────── #
TARGET_CLIPS = 10
WIN_SIZE     = 1.0
MIN_CLIP_LEN = 30
MAX_CLIP_LEN = 90
AUDIO_W      = 2.0
MOTION_W     = 1.0

# ───────── AI MODEL CONFIGURATION ───────── #
MODEL_PATH_STT = "small"
LLM_MODEL_PRIMARY = "meta-llama/Llama-2-7b-chat-hf"
LLM_MODEL_FALLBACK_OPEN = "google/gemma-2b-it"
LLM_MODEL_FALLBACK_SMALL = "distilgpt2" 
MODEL_PATH_VISION_YOLO = "yolov5s.pt" 
# ────────────────────────────────────────── #

# ───────── VIRALITY SCORING CONFIGURATION (Copied from previous state) ───────── #
VIRALITY_CONFIG = {
    'weights': {
        'sentiment_positive': 1.2, 
        'sentiment_negative': 0.8,
        'emotion_joy': 1.2,
        'emotion_surprise': 1.5, 
        'keyword_match': 1.5,  
        'engagement_hook': 1.5,
        'visual_intensity': 1.0,
        'facial_expression_happy': 1.2,
        'fast_cuts_or_action': 1.1,
        'audio_energy_avg': 0.5,
        'motion_avg': 0.3,
        'vertical_cropability': 1.5 
    },
    'thresholds': {'min_sentiment_score': 0.5, 'min_visual_intensity': 0.2,},
    'trending_keywords': ['challenge', 'hack', 'reveal', 'shocking', 'must-see']
}
# ─────────────────────────────────────────────────── #

# ───────── MODEL LOADING FUNCTIONS ───────── #
def load_stt_model():
    global _stt_model
    if WhisperModel is None:
        log.warning("faster_whisper library not available. STT model cannot be loaded. Transcription will be simulated.")
        return

    if _stt_model is None:
        try:
            log.info(f"🧠 Loading STT model: {MODEL_PATH_STT} (will download if not cached)...")
            _stt_model = WhisperModel(
                model_size_or_path=MODEL_PATH_STT,
                device=DEVICE.type, 
                compute_type="int8", 
                local_files_only=False,
            )
            log.info(f"✅ STT Model loaded successfully on [magenta]{DEVICE.type}[/magenta].")
        except Exception as e:
            log.error(f"❌ ERROR loading STT model ({MODEL_PATH_STT}) on {DEVICE.type}: {e}", exc_info=True)
            log.warning("    Ensure 'faster_whisper' is installed correctly and model files can be downloaded/accessed.")
            log.warning("    Transcription will be simulated.")
            _stt_model = None

def load_llm_model():
    global _llm, _tokenizer
    if _llm is not None and _tokenizer is not None:
        log.info("ℹ️ LLM model and tokenizer already loaded.")
        return

    if not all(c in globals() and globals()[c] is not None for c in ['AutoTokenizer', 'AutoModelForCausalLM', 'BitsAndBytesConfig']):
        log.warning("⚠️ Key transformers components (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig) not available. LLM loading skipped.")
        _llm, _tokenizer = None, None
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")
    
    current_quant_config = bnb_config_gpu if DEVICE.type == "cuda" else None

    # Attempt 1: Llama-2 (Primary)
    if hf_token and DEVICE.type == "cuda": 
        log.info(f"🧠 Attempting to load Primary LLM: {LLM_MODEL_PRIMARY} (4-bit with HF_TOKEN on GPU)...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PRIMARY, cache_dir=MODELS_DIR, token=hf_token)
            _llm = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_PRIMARY,
                quantization_config=current_quant_config,
                device_map="auto", 
                cache_dir=MODELS_DIR,
                token=hf_token,
                trust_remote_code=True
            )
            log.info(f"✅ LLM model ({LLM_MODEL_PRIMARY}) and tokenizer loaded successfully on GPU.")
            return
        except Exception as e:
            log.error(f"❌ ERROR loading Primary LLM ({LLM_MODEL_PRIMARY}) on GPU: {e}. Trying fallback.", exc_info=True)
            _llm, _tokenizer = None, None 
    elif hf_token and DEVICE.type == "cpu":
        log.info(f"ℹ️ Primary LLM {LLM_MODEL_PRIMARY} is configured for GPU (4-bit). Skipping on CPU for this model.")
    elif not hf_token:
        log.info("ℹ️ HF_TOKEN not found. Skipping Primary LLM (Llama-2).")

    # Attempt 2: Open Source Fallback
    log.info(f"🧠 Attempting to load Open Fallback LLM: {LLM_MODEL_FALLBACK_OPEN} on [magenta]{DEVICE.type}[/magenta]...")
    try: 
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_FALLBACK_OPEN, cache_dir=MODELS_DIR)
        if DEVICE.type == "cuda":
            log.info(f"   Trying {LLM_MODEL_FALLBACK_OPEN} with 4-bit quantization on GPU...")
            _llm = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_FALLBACK_OPEN,
                quantization_config=current_quant_config, 
                device_map="auto", 
                cache_dir=MODELS_DIR,
                trust_remote_code=True
            )
            log.info(f"✅ LLM model ({LLM_MODEL_FALLBACK_OPEN}) with 4-bit quantization loaded successfully on GPU.")
        else: # CPU
            log.info(f"   Trying {LLM_MODEL_FALLBACK_OPEN} on CPU (no explicit BitsAndBytes quantization)...")
            _llm = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_FALLBACK_OPEN,
                cache_dir=MODELS_DIR,
                trust_remote_code=True
            ).to(DEVICE) 
            log.info(f"✅ LLM model ({LLM_MODEL_FALLBACK_OPEN}) loaded successfully on CPU.")
        return
    except Exception as e_fallback:
        log.error(f"❌ ERROR loading Open Fallback LLM ({LLM_MODEL_FALLBACK_OPEN}) on {DEVICE.type}: {e_fallback}. Trying small fallback.", exc_info=True)
        _llm, _tokenizer = None, None

    # Attempt 3: Small Fallback
    log.info(f"🧠 Attempting to load Small Fallback LLM: {LLM_MODEL_FALLBACK_SMALL} on [magenta]{DEVICE.type}[/magenta]...")
    try:
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_FALLBACK_SMALL, cache_dir=MODELS_DIR)
        _llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_FALLBACK_SMALL,
            cache_dir=MODELS_DIR,
            trust_remote_code=True
        ).to(DEVICE) 
        log.info(f"✅ LLM model ({LLM_MODEL_FALLBACK_SMALL}) and tokenizer loaded successfully on {DEVICE.type}.")
        return
    except Exception as e:
        log.error(f"❌ ERROR loading Small Fallback LLM ({LLM_MODEL_FALLBACK_SMALL}): {e}.", exc_info=True)
        _llm, _tokenizer = None, None

    if _llm is None or _tokenizer is None:
        log.error("❌ All LLM loading attempts failed. Text analysis will be simulated.")
    # Note: _tokenizer and _llm are global variables modified by this function.
    # Gemma-2B-IT ("google/gemma-2b-it") is configured as LLM_MODEL_FALLBACK_OPEN.

def load_vision_models():
    global _yolo_model, _fer_detector
    os.makedirs(MODELS_DIR, exist_ok=True) 

    if _yolo_model is None:
        yolo_weights_path = os.path.join(MODELS_DIR, MODEL_PATH_VISION_YOLO)
        try:
            if os.path.exists(yolo_weights_path):
                log.info(f"🧠 Loading YOLOv5 model from local path: {yolo_weights_path}...")
                _yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights_path, trust_repo=True)
            else:
                log.warning(f"⚠️ YOLOv5 weights '{yolo_weights_path}' not found locally.")
                log.info(f"🧠 Attempting to download and load 'yolov5s' from torch.hub...")
                _yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
            
            _yolo_model.to(DEVICE) 
            log.info(f"✅ YOLOv5 Model loaded successfully on [magenta]{DEVICE.type}[/magenta].")
        except Exception as e:
            log.error(f"❌ ERROR loading YOLOv5 model: {e}", exc_info=True)
            log.warning("YOLOv5 detection will be simulated by placeholder.")
            _yolo_model = YOLOv5Placeholder(weights="N/A", device=DEVICE.type) 

    if FER is not None and _fer_detector is None:
        try:
            # La librería FER actual no acepta parámetro 'device' en __init__
            log.info("🧠  Cargando modelo FER (MTCNN + emotion classifier)...")
            _fer_detector = FER(mtcnn=True)  # ← sin 'device'
            log.info("✅  FER cargado correctamente.")
        except Exception as e:
            log.error(f"❌  Error al cargar FER: {e}", exc_info=True)
            log.warning("El detector de emociones faciales será simulado.")
            _fer_detector = None
# ─────────────────────────────────────────── #

def audio_peaks(video_path: str, win: float) -> np.ndarray:
    """
    Devuelve un vector RMS por ventana `win` (s) a partir del audio del vídeo.
    Si FFmpeg o Librosa fallan, devuelve np.array([]) y registra el error.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        # ── extracción de audio con FFmpeg ───────────────────────────────
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",   # solo errores
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "44100",
            "-c:a", "pcm_s16le",    # WAV PCM 16-bit
            wav_path,
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(
                f"FFmpeg failed (code {result.returncode}).\nSTDERR ↓\n{result.stderr}"
            )
            raise RuntimeError("FFmpeg audio extraction failed")

        # ── cálculo de energía RMS ───────────────────────────────────────
        y, sr = librosa.load(wav_path, sr=None)
        if y.size == 0:
            log.warning("Loaded audio is empty.")
            return np.array([])

        hop = max(int(sr * win), 1)
        rms = librosa.feature.rms(
            y=y,
            frame_length=hop,
            hop_length=hop,
            center=False
        )[0]
        return rms

    except Exception as err:
        log.error(f"ERROR in audio_peaks for {video_path}: {err}", exc_info=True)
        return np.array([])

    finally:
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception as rm_err:
            log.error(f"Cannot remove temp wav file: {rm_err}", exc_info=True)

def _process_frame_buffer(buffer: List[np.ndarray], device: torch.device) -> float:
    if len(buffer) < 2:
        return 0.0

    try:
        # Convert buffer to a NumPy array, then to PyTorch tensor
        # Frames from OpenCV (cv2.read) are HWC, BGR
        batch_np = np.array(buffer, dtype=np.float32) # Ensure float32 for tensor conversion
        batch_tensor = torch.from_numpy(batch_np).to(device) # Shape: (N, H, W, C)

        # Permute to (N, C, H, W) for PyTorch operations
        batch_tensor = batch_tensor.permute(0, 3, 1, 2)

        # Convert BGR to RGB (OpenCV reads as BGR, PyTorch functions often expect RGB)
        # Channel 0 is Blue, 1 is Green, 2 is Red. Flip them.
        batch_tensor_rgb = batch_tensor.flip(dims=[1]) # Flip along the channel dimension
                                                    # This assumes C is at index 1 after permute
                                                    # Correct: permute makes C index 1. BGR (0,1,2) -> RGB (2,1,0)
                                                    # No, flip is not what we need. We need to re-order channels.
                                                    # BGR (0,1,2) -> RGB (2,1,0)
        # Correct BGR to RGB conversion:
        # Create an RGB tensor by selecting channels in the correct order
        # This is more explicit than flip for channel reordering.
        # If batch_tensor is (N, C, H, W) and C is BGR (0,1,2)
        # then red = batch_tensor[:, 2:3, :, :], green = batch_tensor[:, 1:2, :, :], blue = batch_tensor[:, 0:1, :, :]
        # batch_tensor_rgb = torch.cat((batch_tensor[:, 2:3, :, :], batch_tensor[:, 1:2, :, :], batch_tensor[:, 0:1, :, :]), dim=1)
        # Simpler: OpenCV conversion before making tensor
        # For batch, this is tricky. Let's try TF.to_tensor which handles HWC -> CHW and normalizes.
        # However, TF.to_tensor expects PIL image or HWC numpy array.
        # Let's stick to manual conversion for now if TF.rgb_to_grayscale requires RGB.
        # The input to TF.rgb_to_grayscale must be RGB.

        # Assuming batch_tensor is (N,C,H,W) and C is BGR (0,1,2)
        # Convert to RGB: R is channel 2, G is channel 1, B is channel 0
        # This reordering is simpler:
        red_channel = batch_tensor[:, 2, :, :].unsqueeze(1)
        green_channel = batch_tensor[:, 1, :, :].unsqueeze(1)
        blue_channel = batch_tensor[:, 0, :, :].unsqueeze(1)
        batch_tensor_rgb = torch.cat((red_channel, green_channel, blue_channel), dim=1)

        grayscale_batch = TF.rgb_to_grayscale(batch_tensor_rgb) # Shape: (N, 1, H, W)

        # Resize
        resized_batch = TF.resize(grayscale_batch, size=[180, 320], antialias=True) # Using list for size, added antialias
                                                                                # antialias=True is default in newer torchvision

        # Calculate Differences between consecutive frames in the batch
        diffs = torch.abs(resized_batch[1:] - resized_batch[:-1])

        # Sum all pixel differences over all consecutive pairs in the window
        window_motion_score = diffs.sum().item()

        return window_motion_score
    except Exception as e:
        log.error(f"Error processing frame buffer on {device}: {e}", exc_info=True)
        return 0.0

def motion_peaks(video_path: str, win: float) -> tuple[np.ndarray, float, int]:
    """
    Calculates motion peaks from a video file using batch processing on tensors.

    Reads video frames in batches, converts them to grayscale tensors, resizes them,
    and then calculates the sum of absolute differences between consecutive frames
    within each batch (window). This sum represents the motion score for that window.

    Args:
        video_path: Path to the video file.
        win: Duration of the time window (in seconds) for which to calculate
             a single motion score.

    Returns:
        A tuple containing:
            - np.ndarray: An array of motion scores, one for each processed window.
            - float: The core processing time (in seconds) spent in the frame reading
                     and processing loop.
            - int: The total number of frames read and considered for processing from the video.
    """
    # DEVICE is globally defined
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Failed to open video: [cyan]{video_path}[/cyan] in motion_peaks.")
        return np.array([], dtype=np.float32), 0.0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        log.warning(f"Could not get FPS from video [cyan]{video_path}[/cyan]. Motion analysis might be incorrect or fail.")
        cap.release()
        return np.array([], dtype=np.float32), 0.0, 0

    frames_per_window = max(2, int(fps * win)) # Ensure at least 2 frames for difference calculation
    log.info(f"Motion analysis: Video FPS: {fps:.2f}, Window size: {win}s, Frames per window: {frames_per_window}")

    mvals = []
    frame_buffer: List[np.ndarray] = []
    total_frames_read_for_processing = 0
    
    loop_start_time = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret: # End of video
            if len(frame_buffer) >= 2: # Process final partial window if it has at least 2 frames
                log.info(f"Processing final partial window of {len(frame_buffer)} frames for motion at end of video.")
                motion_score = _process_frame_buffer(frame_buffer, DEVICE)
                mvals.append(motion_score)
            else:
                log.info(f"Final partial window has {len(frame_buffer)} frames, less than 2. Skipping.")
            break

        frame_buffer.append(frame)
        # frame_count += 1 # Not used

        if len(frame_buffer) == frames_per_window:
            motion_score = _process_frame_buffer(frame_buffer, DEVICE)
            mvals.append(motion_score)
            # Clear buffer for the next set of frames.
            # Important: If overlap is desired, this needs to be smarter, e.g., keep last N-1 frames.
            # For non-overlapping windows as implied by original `step` logic:
            frame_buffer = []
            # If overlapping windows are desired, the logic should be:
            # frame_buffer = frame_buffer[frames_to_slide:] or similar.
            # The current implementation implies non-overlapping windows of `frames_per_window`.
        else: # Frame read successfully
            total_frames_read_for_processing +=1


    loop_duration = time.perf_counter() - loop_start_time
    cap.release()

    if not mvals:
        log.warning(f"No motion values calculated for video {video_path}. This might be due to very short video or processing issues.")
        return np.array([], dtype=np.float32), loop_duration, total_frames_read_for_processing

    return np.array(mvals, dtype=np.float32), loop_duration, total_frames_read_for_processing


def transcribe_video(video_path: str) -> List[Dict]:
    """
    Extrae el audio con FFmpeg, lo pasa al modelo Whisper (faster-whisper)
    y devuelve la transcripción en una lista de diccionarios:
    [{start_time, end_time, text}, …]

    Si falla FFmpeg o el modelo, devuelve una transcripción simulada.
    """
    # ── fallback por si el modelo no está cargado ────────────────
    if _stt_model is None:
        log.warning(
            f"STT model not loaded; transcription will be simulated "
            f"para «{video_path}»"
        )
        return [
            {
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "Simulated: STT model not available.",
            }
        ]

    log.info(f"🎙️  Actual STT: transcribing [cyan]{video_path}[/cyan]…")

    # Ruta temporal WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        audio_output_path = tmp_audio.name

    try:
        # ---------- 1) Extraer audio con FFmpeg ----------
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",          # solo errores (menos ruido)
            "-y",
            "-i",
            video_path,
            "-vn",            # sin vídeo
            "-ac",
            "1",              # mono
            "-ar",
            "16000",          # 16 kHz
            "-c:a",
            "pcm_s16le",      # WAV PCM 16-bit
            audio_output_path,
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(
                f"FFmpeg failed (code {result.returncode}).\n"
                f"STDERR ↓\n{result.stderr}"
            )
            raise RuntimeError("FFmpeg audio extraction failed")

        # ---------- 2) Transcribir con faster-whisper ----------
        segments_gen, info = _stt_model.transcribe(audio_output_path, beam_size=5, word_timestamps=True)
        log.info(
            f"Detected language '[italic yellow]{info.language}[/italic yellow]' "
            f"prob={info.language_probability:.2f}, Word-level timestamps enabled."
        )

        transcript = []
        for seg in segments_gen:
            segment_data = {
                "start_time": round(seg.start, 2),
                "end_time": round(seg.end, 2),
                "text": seg.text.strip(),
            }
            if hasattr(seg, 'words') and seg.words:
                segment_data['words'] = [
                    {'word': word.word, 'start': round(word.start, 2), 'end': round(word.end, 2)}
                    for word in seg.words
                ]
            transcript.append(segment_data)

        log.info("✅  Actual STT: transcription complete.")
        return transcript

    except Exception as err:
        log.error(
            f"ERROR during STT transcription for [cyan]{video_path}[/cyan]: {err}",
            exc_info=True,
        )
        log.warning("Falling back to simulated transcription.")
        return [
            {
                "start_time": 0.0,
                "end_time": 5.0,
                "text": f"Simulated: STT error ({err}).",
            }
        ]
    finally:
        # Limpieza del WAV temporal
        try:
            if os.path.exists(audio_output_path):
                os.remove(audio_output_path)
        except Exception as rm_err:
            log.error(f"Could not remove temp audio file: {rm_err}", exc_info=True)


def analyze_text(llm_segment_info: Dict, timestamp_info: Dict) -> Dict:
    """
    Processes information from a pre-analyzed LLM-suggested segment to fit
    the structure expected by the `calculate_virality_score` function.

    This function no longer makes direct LLM calls. Instead, it adapts the
    `theme` and `reason_for_highlight` provided by `get_llm_candidate_segments`
    into a format usable for scoring (e.g., using theme as a proxy for emotion
    and extracting keywords from the reason).

    Args:
        llm_segment_info: A dictionary representing one segment suggestion from
                          `get_llm_candidate_segments`. Expected keys include
                          'theme' and 'reason_for_highlight'.
        timestamp_info: A dictionary with 'start' and 'end' times for the
                        specific clip being processed (which might have been
                        adjusted from the original LLM suggestion).

    Returns:
        A dictionary structured for use in `calculate_virality_score`, containing
        keys like 'timestamp_info', 'sentiment', 'emotions', 'keywords',
        'raw_llm_output' (now a summary of LLM suggestion), and 'summary'.
    """
    theme = llm_segment_info.get('theme', 'unknown_theme')
    reason = llm_segment_info.get('reason_for_highlight', '')

    # For keywords, we can split the reason or theme.
    # Taking first 5 words from reason, or theme if reason is short.
    keywords_source = reason if len(reason.split()) > 2 else theme + " " + reason
    keywords = keywords_source.lower().split()[:5]

    # Sentiment and emotions are now more directly derived or placeholders
    # as the detailed per-segment LLM call is removed.
    # We can assign a generic sentiment or try to infer it from theme/reason if needed later.
    # For now, let's use a neutral sentiment and theme as emotion.
    return {
        'timestamp_info': timestamp_info, # This should be the actual clip's timestamp
        'sentiment': {'label': 'neutral_from_llm_segment', 'score': 0.5}, # Placeholder sentiment
        'emotions': [theme.lower().replace(" ", "_")], # Use theme as a proxy for emotion
        'keywords': keywords,
        'raw_llm_output': f"Theme: {theme}, Reason: {reason}", # Store the source info
        'summary': f"Derived from LLM segment suggestion: {theme}",
        # Include original llm segment data for reference if needed downstream
        'llm_segment_suggestion': llm_segment_info
    }

def analyze_visuals(full_video_path: str, timestamp_info: Dict, frames_to_process: int = 3) -> Dict:
    simulated_yolo = [{'object': 'simulated_object', 'confidence': 0.0, 'bbox': [0,0,0,0]}]
    simulated_fer = [{'box': [0,0,0,0], 'emotions': {'neutral_simulated': 1.0}}]
    default_frame_dims = {'width': 1920, 'height': 1080} 
    default_response = {
        'timestamp_info': timestamp_info,
        'key_objects': simulated_yolo,
        'facial_expressions': simulated_fer,
        'frame_dimensions': default_frame_dims, 
        'visual_intensity_score': 0.1, 
        'notes': "Visual analysis simulated: Models not loaded or error during processing."
    }

    if (_yolo_model is None or isinstance(_yolo_model, YOLOv5Placeholder)) and \
       (_fer_detector is None):
        log.warning(f"Vision models (YOLO & FER) not loaded for visual analysis of [cyan]{full_video_path}[/cyan]. Using placeholders.")
        return default_response

    cap = cv2.VideoCapture(full_video_path)
    if not cap.isOpened():
        log.error(f"Could not open video file [cyan]{full_video_path}[/cyan] for visual analysis.")
        default_response['notes'] = f"Error: Could not open video {full_video_path}."
        return default_response

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: 
        log.warning(f"Video FPS is 0 for [cyan]{full_video_path}[/cyan]. Attempting to use a default FPS of 25, but timing might be inaccurate.")
        fps = 25 

    segment_start_time = timestamp_info['start']
    segment_end_time = timestamp_info['end']
    segment_duration = segment_end_time - segment_start_time

    processed_frames_yolo = []
    processed_frames_fer = []
    frame_width, frame_height = default_frame_dims['width'], default_frame_dims['height'] 
    
    frames_captured_for_segment = 0
    first_processed_frame_dims_captured = False

    try:
        frame_indices_to_sample = []
        if segment_duration > 0 and frames_to_process > 0:
            time_points = np.linspace(segment_start_time, segment_end_time, frames_to_process)
            frame_indices_to_sample = [int(t * fps) for t in time_points]
        
        current_frame_idx = 0
        
        if frame_indices_to_sample:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break 
                
                if current_frame_idx in frame_indices_to_sample:
                    frames_captured_for_segment +=1
                    
                    if not first_processed_frame_dims_captured:
                        h_cap, w_cap = frame.shape[:2]
                        if h_cap > 0 and w_cap > 0: 
                            frame_height, frame_width = h_cap, w_cap
                            first_processed_frame_dims_captured = True

                    if _yolo_model is not None and not isinstance(_yolo_model, YOLOv5Placeholder):
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                        results = _yolo_model(frame_rgb)
                        for det in results.xyxy[0].cpu().numpy(): 
                            processed_frames_yolo.append({
                                'class': _yolo_model.names[int(det[5])],
                                'confidence': float(det[4]),
                                'bbox': [float(c) for c in det[:4]] 
                            })
                    
                    if _fer_detector is not None:
                        emotions_in_frame = _fer_detector.detect_emotions(frame)
                        for face_emotion in emotions_in_frame:
                            processed_frames_fer.append({
                                'box': face_emotion['box'], 
                                'emotions': face_emotion['emotions'] 
                            })
                
                current_frame_idx += 1
                if frames_captured_for_segment >= len(frame_indices_to_sample) and len(frame_indices_to_sample) > 0 :
                    break 
                if current_frame_idx > max(frame_indices_to_sample, default=0) + int(fps) and len(frame_indices_to_sample) > 0: 
                    log.warning(f"Safety break in frame processing for [cyan]{full_video_path}[/cyan] at segment {timestamp_info}")
                    break
    except Exception as e:
        log.error(f"ERROR during visual analysis frame processing for [cyan]{full_video_path}[/cyan]: {e}", exc_info=True)
        default_response['notes'] = f"Error during visual frame processing: {e}"
    finally:
        cap.release()

    final_yolo_results = processed_frames_yolo if processed_frames_yolo else simulated_yolo
    final_fer_results = processed_frames_fer if processed_frames_fer else simulated_fer
    
    notes = "Visual analysis complete."
    if not processed_frames_yolo and (_yolo_model is None or isinstance(_yolo_model, YOLOv5Placeholder)):
        notes += " YOLO not run (model not loaded)."
    elif not processed_frames_yolo:
        notes += " No objects detected by YOLO or no frames processed."
        
    if not processed_frames_fer and _fer_detector is None:
        notes += " FER not run (model not loaded)."
    elif not processed_frames_fer:
        notes += " No faces detected by FER or no frames processed."

    if not frames_captured_for_segment and frame_indices_to_sample :
         notes += " No frames were actually captured or processed for the segment."

    return {
        'timestamp_info': timestamp_info,
        'key_objects': final_yolo_results,
        'facial_expressions': final_fer_results,
        'frame_dimensions': {'width': frame_width, 'height': frame_height} if first_processed_frame_dims_captured else default_frame_dims,
        'visual_intensity_score': 0.5 if (processed_frames_yolo or processed_frames_fer) else 0.1, 
        'notes': notes
    }

async def get_llm_candidate_segments(full_transcript: List[Dict], video_duration: float) -> List[Dict]:
    """
    Identifies potential highlight segments from the full video transcript using an LLM.

    This function first estimates the token count of the entire transcript.
    If it's within `MAX_TOKENS_FOR_LLM`, it calls the LLM once with the full transcript.
    If it exceeds the limit, the transcript is split into overlapping chunks using
    `_split_transcript_into_chunks`. Each chunk is then processed by the LLM
    concurrently using `asyncio.gather` with `_call_llm_for_segments`.
    Finally, if chunking was used, results are deduplicated based on start time proximity.

    Args:
        full_transcript: A list of dictionaries representing the full video transcript
                         (e.g., output from `transcribe_video`).
        video_duration: The total duration of the video in seconds.

    Returns:
        A list of candidate segment dictionaries suggested by the LLM, potentially
        deduplicated if chunking was performed. Returns an empty list on failure
        or if no candidates are found.
    """
    if not _llm or not _tokenizer:
        log.warning("LLM model or tokenizer not loaded. Cannot get LLM candidate segments.")
        return []

    if not full_transcript:
        log.warning("Full transcript is empty. Cannot get LLM candidate segments.")
        return []

    MAX_TOKENS_FOR_LLM = 7500  # Max tokens for model context (Gemma-2B-IT is 8192, leaving buffer)
    MAX_TOKENS_PER_CHUNK = 6000 # Target for each chunk if splitting
    OVERLAP_SEGMENTS_COUNT = 7 # Number of transcript segments to overlap between chunks

    transcript_text_for_estimation = " ".join([segment['text'] for segment in full_transcript if 'text' in segment])
    if not transcript_text_for_estimation.strip():
        log.warning("Transcript text is blank after concatenation. Cannot get LLM candidate segments.")
        return []

    estimated_tokens = len(_tokenizer.encode(transcript_text_for_estimation))
    log.info(f"Estimated token count for full transcript: {estimated_tokens}")

    all_candidate_segments = []

    if estimated_tokens <= MAX_TOKENS_FOR_LLM:
        log.info("Transcript within token limit, processing as a single call.")
        all_candidate_segments = await _call_llm_for_segments(full_transcript, video_duration, is_chunk=False)
    else:
        log.info(f"Transcript exceeds token limit ({estimated_tokens} > {MAX_TOKENS_FOR_LLM}). Splitting into chunks.")
        transcript_chunks = _split_transcript_into_chunks(
            full_transcript,
            _tokenizer,
            max_tokens_per_chunk=MAX_TOKENS_PER_CHUNK,
            overlap_segments_count=OVERLAP_SEGMENTS_COUNT
        )
        log.info(f"Split transcript into {len(transcript_chunks)} chunks.")

        tasks = []
        for i, chunk_segments in enumerate(transcript_chunks):
            tasks.append(
                _call_llm_for_segments(
                    chunk_segments,
                    video_duration,
                    is_chunk=True,
                    chunk_num=i+1,
                    total_chunks=len(transcript_chunks)
                )
            )

        if tasks:
            log.info(f"Dispatching {len(tasks)} LLM calls concurrently for transcript chunks...")
            # Run all chunk processing tasks concurrently using asyncio.gather
            results_from_chunks = await asyncio.gather(*tasks)
            log.info("All concurrent LLM calls for chunks completed.")
            for chunk_result in results_from_chunks:
                if chunk_result: # chunk_result is a list of segments
                    all_candidate_segments.extend(chunk_result)

        if all_candidate_segments:
            log.info(f"Collected {len(all_candidate_segments)} candidates from all chunks before deduplication.")
            # Deduplication
            # Sort by start time, then by end time as a secondary factor.
            all_candidate_segments.sort(key=lambda x: (x.get('start_time', 0), x.get('end_time', 0)))

            deduplicated_segments = []
            if not all_candidate_segments: return [] # Should not happen if we got here

            deduplicated_segments.append(all_candidate_segments[0])
            for current_segment in all_candidate_segments[1:]:
                prev_segment = deduplicated_segments[-1]
                # Simple deduplication: if start times are very close (e.g., within 5s)
                # and themes are similar (optional, could be too strict)
                # This primarily handles direct overlaps from chunking.
                # A more sophisticated IoU (Intersection over Union) could be used.
                # For now, a simple start time proximity is a good first pass.
                if current_segment.get('start_time', 0) > prev_segment.get('start_time', 0) + 5.0 : # 5s threshold
                    deduplicated_segments.append(current_segment)
                else:
                    # If start times are close, prefer the one with a (slightly) longer duration or perhaps higher confidence if LLM provided it
                    # For now, just keeping the first one encountered in sorted list if they are too close.
                    # Or, if themes are very different, might still keep it.
                    # Let's refine: if start times are close, check if it's essentially the same segment.
                    # A simple check: if start_time is within X seconds and end_time is also within Y seconds.
                    if abs(current_segment.get('start_time', 0) - prev_segment.get('start_time', 0)) < 5.0 and \
                       abs(current_segment.get('end_time', 0) - prev_segment.get('end_time', 0)) < 10.0:
                        log.debug(f"Deduplicating segment: Current {current_segment.get('start_time')} Theme: '{current_segment.get('theme')}' with Prev {prev_segment.get('start_time')} Theme: '{prev_segment.get('theme')}'")
                        # Potentially merge reasons or choose the one with a more detailed reason, for now just skip current.
                        continue
                    else:
                         deduplicated_segments.append(current_segment)


            log.info(f"Reduced to {len(deduplicated_segments)} candidates after deduplication.")
            all_candidate_segments = deduplicated_segments

    return all_candidate_segments

def _split_transcript_into_chunks(
    full_transcript: List[Dict],
    tokenizer: AutoTokenizer,
    max_tokens_per_chunk: int,
    overlap_segments_count: int
) -> List[List[Dict]]:
    """
    Splits the full_transcript (list of Whisper segment dicts) into overlapping chunks.

    Each chunk is designed to have a token count (based on concatenated segment text)
    that is less than `max_tokens_per_chunk`, considering an estimated overhead for
    the LLM prompt. Chunks overlap by `overlap_segments_count` segments to ensure
    context continuity for the LLM.

    Args:
        full_transcript: The complete list of transcript segments.
        tokenizer: The Hugging Face tokenizer used for estimating token counts.
        max_tokens_per_chunk: The target maximum token count for the text content
                              of each chunk (excluding prompt template overhead).
        overlap_segments_count: The number of transcript segments from the end of
                                the previous chunk to include at the beginning of
                                the next chunk.

    Returns:
        A list of chunks, where each chunk is itself a list of transcript segment
        dictionaries.
    """
    chunks = []
    current_chunk_segments: List[Dict] = []
    current_chunk_tokens = 0

    # Estimate prompt overhead (template text without transcript itself).
    # This is a rough estimate; the actual token count of the constructed prompt might vary slightly.
    dummy_prompt_template_for_overhead = f"""You are an expert video editor... (rest of prompt template without actual transcript text)..."""
    prompt_overhead_tokens = len(tokenizer.encode(dummy_prompt_template_for_overhead))

    # Calculate the effective maximum tokens available for the transcript text within each chunk.
    # A buffer is subtracted for the expected JSON response format and other variations.
    effective_max_tokens_for_text = max_tokens_per_chunk - prompt_overhead_tokens - 200

    segment_idx = 0
    while segment_idx < len(full_transcript):
        current_segment = full_transcript[segment_idx]
        current_segment_text = current_segment.get('text', '')
        current_segment_tokens = len(tokenizer.encode(current_segment_text))

        # If the current chunk is empty or adding the next segment does not exceed the token limit
        if not current_chunk_segments or (current_chunk_tokens + current_segment_tokens <= effective_max_tokens_for_text):
            current_chunk_segments.append(current_segment)
            current_chunk_tokens += current_segment_tokens
            segment_idx += 1
        else:
            # The current chunk is full, or the next segment would make it too large.
            # Add the current_chunk_segments to the list of chunks.
            if current_chunk_segments: # Ensure there's something to add
                 chunks.append(list(current_chunk_segments)) # Add a copy

            # Determine the starting point for the next chunk to create overlap.
            # The goal is to re-include `overlap_segments_count` from the *end* of the chunk just added.
            # `segment_idx` is currently pointing to the segment that *didn't* fit.
            # The last segment added to `current_chunk_segments` was at `segment_idx - 1`.
            # So, the overlap should start `overlap_segments_count` segments before `segment_idx -1`.
            # However, it's simpler to think: the new chunk starts `overlap_segments_count` segments
            # *before* the segment that caused the current chunk to be finalized.
            # If `current_chunk_segments` has `k` items, its last item was `full_transcript[segment_idx-1]`.
            # The new chunk should effectively start from index `(segment_idx - 1) - overlap_segments_count + 1` if we consider the items.
            # Or, more directly, the current `segment_idx` is the one that couldn't be added.
            # We want the new chunk to start `overlap_segments_count` segments *before* this current `segment_idx`.
            # This means the next loop iteration for `segment_idx` should start from there.

            # Corrected overlap logic:
            # The last segment included in the *previous* chunk is at index `segment_idx - 1`.
            # To overlap, the new chunk should start `overlap_segments_count` segments *before* the *end* of the previous chunk.
            # If the previous chunk had `N` segments, ending at `segment_idx - 1`,
            # its first segment was at `segment_idx - N`.
            # The new chunk should start from `(segment_idx - 1) - (overlap_segments_count - 1)`.
            # This makes `segment_idx` for the next iteration.
            if current_chunk_segments: # Ensure current_chunk_segments was not empty
                 # Reset segment_idx to start of the overlap for the next chunk.
                 # The last segment added to the previous chunk was at original index `segment_idx - 1`.
                 # To get an overlap of `overlap_segments_count`, the new chunk should start at
                 # `(segment_idx - 1) - overlap_segments_count + 1`.
                 # Example: if overlap is 5, and last added was index 20, new chunk starts at 20-5+1 = 16.
                 # `segment_idx` will then be correctly incremented from this new starting point.
                start_of_last_added_segment_in_full_transcript = full_transcript.index(current_chunk_segments[0]) + len(current_chunk_segments) -1
                segment_idx = max(0, start_of_last_added_segment_in_full_transcript - overlap_segments_count + 1)


            current_chunk_segments = [] # Reset for the new chunk
            current_chunk_tokens = 0
            # The loop continues, and segment_idx will now point to the start of the overlapping segment.

    # Add any remaining segments as the last chunk
    if current_chunk_segments:
        chunks.append(list(current_chunk_segments))

    return chunks


async def _call_llm_for_segments(transcript_segments: List[Dict], video_duration: float, is_chunk: bool = False, chunk_num: int = 0, total_chunks: int = 0) -> List[Dict]:
    """
    Asynchronously calls the LLM to get candidate segments from transcript text.

    This function prepares a prompt with the given transcript segments (which can be
    a full transcript or a chunk thereof), invokes the LLM using a synchronous
    Hugging Face pipeline run in a separate thread (via `asyncio.to_thread`),
    and then parses the LLM's JSON output.

    Args:
        transcript_segments: A list of dictionaries, where each dictionary represents
                             a segment of the transcript (e.g., from Whisper).
                             Expected keys: 'text'.
        video_duration: The total duration of the video in seconds, used for context in the prompt.
        is_chunk: Boolean flag indicating if the `transcript_segments` represent a chunk
                  of a larger transcript. Defaults to False.
        chunk_num: If `is_chunk` is True, the number of the current chunk.
        total_chunks: If `is_chunk` is True, the total number of chunks.

    Returns:
        A list of dictionaries, where each dictionary represents a candidate highlight
        segment suggested by the LLM. Returns an empty list if errors occur or no
        valid segments are found.
    """
    if not _llm or not _tokenizer:
        log.warning("LLM model or tokenizer not loaded in _call_llm_for_segments.")
        return []

    transcript_text = " ".join([segment['text'] for segment in transcript_segments if 'text' in segment])
    if not transcript_text.strip():
        log.warning("Transcript text for LLM call is blank.")
        return []

    prompt_intro = "You are an expert video editor tasked with identifying potentially viral highlight clips from the following video transcript."
    if is_chunk:
        prompt_intro = f"You are an expert video editor. You are analyzing CHUNK {chunk_num}/{total_chunks} of a video transcript. Focus on identifying viral moments within this specific chunk. All timestamps are absolute from the beginning of the full video."

    # Max new tokens should be dynamically adjusted based on remaining context or typical response size.
    # For now, keeping it fixed as it was.
    max_new_tokens_for_llm_response = 3000

    prompt = f"""{prompt_intro}
The total video duration is {video_duration:.2f} seconds.
Your goal is to identify around 15-20 distinct segments (or fewer if the chunk is short) that have high potential to be engaging short clips.
Each segment should ideally be between 30 and 90 seconds long.

Provide your output as a JSON list of dictionaries. Each dictionary in the list should represent one candidate segment and must follow this exact format:
{{
  "start_time": <float>,  // Start time of the segment in seconds from the beginning of the video
  "end_time": <float>,    // End time of the segment in seconds from the beginning of the video
  "theme": "<string>",     // A concise theme or topic for this segment (e.g., "Funny Reaction", "Key Product Feature", "Dramatic Climax")
  "reason_for_highlight": "<string>", // Brief explanation why this segment is a good candidate for a highlight
  "estimated_duration": <float> // Estimated duration of the segment in seconds (end_time - start_time)
}}

Ensure that "start_time" and "end_time" are accurate floating-point numbers representing seconds from the video's beginning.
Ensure "estimated_duration" is correctly calculated.
Do not include any segments that would be shorter than 10 seconds or longer than 120 seconds, aiming for the 30-90 second range.

Here is the video transcript (or a chunk of it):
--- START TRANSCRIPT ---
{transcript_text}
--- END TRANSCRIPT ---

Now, provide the JSON list of candidate segments:
"""
    source_log_str = f"Source: {'Chunk ' + str(chunk_num) + '/' + str(total_chunks) if is_chunk else 'Full Transcript'}"
    log.info(f"🧠 Requesting candidate segments from LLM. {source_log_str}. Text length: {len(transcript_text)} chars.")

    llm_raw_output = ""
    try:
        # Synchronous part to be run in a thread
        def sync_llm_call(p_text, pipeline_obj, max_tokens):
            # This pipeline object might not be thread-safe if it shares underlying resources like model weights in a mutable way.
            # However, Hugging Face pipelines are generally designed to be used in such scenarios,
            # and issues usually arise from GPU memory contention if multiple threads try to use the GPU simultaneously without proper management.
            # For CPU-bound models or if device_map="auto" handles this well, it might be fine.
            # If issues arise, a process pool or a more sophisticated queuing mechanism might be needed.
            outputs = pipeline_obj(p_text, max_new_tokens=max_tokens, do_sample=False, temperature=0.1, top_k=5) # Reduced top_k from previous state
            if outputs and isinstance(outputs, list) and "generated_text" in outputs[0]:
                return outputs[0]["generated_text"]
            return None

        # Create the pipeline object inside the async function, but before passing to thread.
        # This ensures that the pipeline object is created in the context of the event loop,
        # though the actual execution happens in the thread.
        # For some models/pipelines, this might still be an issue if the model itself is not thread-safe.
        # If _llm and _tokenizer are global and loaded on the main thread, this should generally be okay.
        text_gen_pipeline_obj = pipeline("text-generation", model=_llm, tokenizer=_tokenizer)

        llm_raw_output = await asyncio.to_thread(
            sync_llm_call,
            prompt,
            text_gen_pipeline_obj, # Pass the created pipeline object
            max_new_tokens_for_llm_response
        )

        if not llm_raw_output: # Check if llm_raw_output is None or empty
            log.error(f"❌ LLM pipeline ({source_log_str}) did not return any text output.")
            return []

        if prompt.strip() in llm_raw_output: # Strip prompt from beginning
            llm_raw_output = llm_raw_output.split(prompt.strip(), 1)[-1].strip()
        elif "Now, provide the JSON list of candidate segments:" in llm_raw_output:
            llm_raw_output = llm_raw_output.split("Now, provide the JSON list of candidate segments:", 1)[-1].strip()

        log.debug(f"LLM Raw Output for {source_log_str}:\n{llm_raw_output}")

        json_start_index = llm_raw_output.find('[')
        json_end_index = llm_raw_output.rfind(']')

        if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
            json_str = llm_raw_output[json_start_index : json_end_index + 1]
            try:
                candidate_segments = json.loads(json_str)
                if isinstance(candidate_segments, list):
                    valid_segments = []
                    for item in candidate_segments:
                        if isinstance(item, dict) and \
                           all(key in item for key in ["start_time", "end_time", "theme", "reason_for_highlight", "estimated_duration"]) and \
                           isinstance(item["start_time"], (int, float)) and \
                           isinstance(item["end_time"], (int, float)) and \
                           isinstance(item["estimated_duration"], (int, float)):
                            if not (0 <= item["start_time"] < video_duration and 0 < item["end_time"] <= video_duration + 0.1): # allow for slight overrun due to rounding
                                log.warning(f"LLM suggested segment with out-of-bounds times for video duration {video_duration:.2f}s: {item}. ({source_log_str}). Skipping.")
                                continue
                            valid_segments.append(item)
                        else:
                            log.warning(f"Skipping invalid segment from LLM ({source_log_str}): {item}")
                    log.info(f"✅ LLM ({source_log_str}) returned {len(valid_segments)} valid candidate segments.")
                    return valid_segments
                else:
                    log.error(f"❌ LLM output ({source_log_str}) was valid JSON, but not a list. Output: {json_str}")
                    return []
            except json.JSONDecodeError as e_json:
                log.error(f"❌ Failed to parse JSON from LLM output ({source_log_str}): {e_json}", exc_info=False)
                log.debug(f"Problematic JSON string ({source_log_str}): {json_str}")
                return []
        else:
            log.error(f"❌ Could not find a valid JSON list structure in LLM output ({source_log_str}). Raw output was: {llm_raw_output}")
            return []

    except Exception as e:
        log.error(f"❌ ERROR during LLM call ({source_log_str}): {e}", exc_info=True)
        log.debug(f"LLM Raw Output at time of error ({source_log_str}): {llm_raw_output}")
        return []

def calculate_virality_score(
    text_analysis: Dict, visual_analysis: Dict,
    segment_audio_avg_rms: float, segment_motion_avg_score: float,
    config: Dict = VIRALITY_CONFIG
) -> float:
    score = 0.0; weights = config.get('weights', {}); thresholds = config.get('thresholds', {}); trending_keywords = config.get('trending_keywords', [])
    if text_analysis:
        sentiment_label = text_analysis.get('sentiment', {}).get('label', 'neutral'); sentiment_score_val = text_analysis.get('sentiment', {}).get('score', 0.0)
        if sentiment_score_val > thresholds.get('min_sentiment_score', 0.3):
            if sentiment_label == 'positive': score += sentiment_score_val * weights.get('sentiment_positive', 1.0)
            elif sentiment_label == 'negative': score += sentiment_score_val * weights.get('sentiment_negative', 0.5)
        for emotion in text_analysis.get('emotions', []): score += weights.get(f'emotion_{emotion}', 0.0)
        found_keywords = text_analysis.get('keywords', []);
        for kw in found_keywords:
            if kw in trending_keywords: score += weights.get('keyword_match', 1.0)
        if text_analysis.get('engagement_hooks'): score += weights.get('engagement_hook', 1.0)
    if visual_analysis:
        visual_intensity = visual_analysis.get('visual_intensity_score', 0.0)
        if visual_intensity > thresholds.get('min_visual_intensity', 0.1): score += visual_intensity * weights.get('visual_intensity', 1.0)
        
        for face in visual_analysis.get('facial_expressions', []):
            if isinstance(face, dict) and 'emotions' in face:
                if face['emotions'].get('happy', 0.0) > 0.5 or face['emotions'].get('surprise', 0.0) > 0.3: 
                     score += weights.get('facial_expression_happy', 1.0) 
            elif isinstance(face, dict) and face.get('expression') == 'happy': 
                 score += weights.get('facial_expression_happy', 1.0)

        if len(visual_analysis.get('scene_changes', [])) > 2 : score += weights.get('fast_cuts_or_action', 1.0) 

        vertical_cropability_score = 0.0
        key_objects = visual_analysis.get('key_objects', [])
        frame_dims = visual_analysis.get('frame_dimensions')

        if frame_dims and frame_dims.get('width', 0) > 0 and frame_dims.get('height', 0) > 0:
            frame_width = frame_dims['width']
            frame_height = frame_dims['height']
            
            main_subject = None
            persons = [obj for obj in key_objects if obj.get('class') == 'person' and obj.get('confidence', 0) > 0.3]
            if persons:
                persons.sort(key=lambda p: (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]), reverse=True) 
                main_subject = persons[0]
            elif key_objects and key_objects[0].get('confidence',0) > 0: 
                valid_objects = [obj for obj in key_objects if obj.get('confidence', 0) > 0.0 and 'simulated' not in obj.get('class','')]
                if valid_objects:
                    valid_objects.sort(key=lambda p: (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]), reverse=True)
                    main_subject = valid_objects[0]

            if main_subject:
                x1, y1, x2, y2 = main_subject['bbox']
                subj_center_x = (x1 + x2) / 2
                
                target_aspect_ratio = 9/16
                current_aspect_ratio = frame_width / frame_height

                if current_aspect_ratio > target_aspect_ratio: 
                    crop_height = frame_height
                    crop_width = crop_height * target_aspect_ratio
                else: 
                    crop_width = frame_width
                    crop_height = crop_width / target_aspect_ratio 
                
                crop_x1 = subj_center_x - crop_width / 2
                crop_x2 = subj_center_x + crop_width / 2
                
                crop_x1 = max(0, crop_x1)
                crop_x2 = min(frame_width, crop_x2)

                overlap_x1 = max(x1, crop_x1)
                overlap_x2 = min(x2, crop_x2)
                
                overlap_width = max(0, overlap_x2 - overlap_x1)
                subject_width = x2 - x1
                
                if subject_width > 0:
                    coverage = overlap_width / subject_width
                    vertical_cropability_score = coverage

            score += vertical_cropability_score * weights.get('vertical_cropability', 1.5)
        else:
            if not frame_dims or frame_dims.get('width',0) == 0 : log.warning("Frame dimensions not available for cropability score calculation.")

    score += segment_audio_avg_rms * weights.get('audio_energy_avg', 0.1) 
    score += segment_motion_avg_score * weights.get('motion_avg', 0.1) 
    return round(score, 2)

def combined_score( 
    audio_energy: np.ndarray, motion_score: np.ndarray, aw: float, mw: float,
    text_analysis_results: Dict = None, visual_analysis_results: Dict = None
) -> np.ndarray:
    a = (audio_energy  - audio_energy.mean())  / (audio_energy.std()  + 1e-6)
    m = (motion_score - motion_score.mean()) / (motion_score.std() + 1e-6)
    n = min(len(a), len(m))
    current_score = aw * a[:n] + mw * m[:n]
    return current_score

def get_transcript_for_window(full_transcript: List[Dict], window_start_time: float, window_end_time: float) -> str:
    segment_texts = []
    for entry in full_transcript:
        if entry['start_time'] < window_end_time and entry['end_time'] > window_start_time:
            segment_texts.append(entry['text'])
    return " ".join(segment_texts)

async def best_segments(
    video_path: str, full_transcript: List[Dict],
    audio_energy_per_second: np.ndarray, motion_score_per_second: np.ndarray,
    win_size: float, nclips: int = TARGET_CLIPS,
    min_len: int = MIN_CLIP_LEN, max_len: int = MAX_CLIP_LEN
) -> List[Dict]:
    """
    Identifies the best video segments based on LLM suggestions and multi-modal analysis.

    This asynchronous function orchestrates the core logic for selecting highlight clips.
    It first determines the video duration. Then, it calls `get_llm_candidate_segments`
    (which itself might perform transcript chunking and parallel LLM calls) to get
    initial segment suggestions.

    Each LLM-suggested segment is then processed:
    1.  Times are validated and adjusted to fit MIN_CLIP_LEN and MAX_CLIP_LEN.
    2.  Text analysis data is derived using the (new) `analyze_text` from LLM's theme/reason.
    3.  Visual analysis is performed using `analyze_visuals`.
    4.  Average audio energy and motion scores are calculated for the segment.
    5.  A `virality_score` is computed.

    Finally, segments are sorted by score, and the top N non-overlapping clips
    are selected.

    Args:
        video_path: Path to the video file.
        full_transcript: The complete transcript of the video.
        audio_energy_per_second: NumPy array of RMS audio energy per second.
        motion_score_per_second: NumPy array of motion scores per second.
        win_size: The window size (in seconds) used for audio/motion analysis.
        nclips: The target number of clips to select.
        min_len: Minimum length of a clip in seconds.
        max_len: Maximum length of a clip in seconds.

    Returns:
        A list of dictionaries, where each dictionary represents a selected highlight
        clip with its start time, end time, score, and duration.
    """
    if not audio_energy_per_second.any() : log.warning("Audio energy data is empty for segment scoring.");
    # --- Determine video_duration_seconds ---
    video_duration_seconds = 0
    if audio_energy_per_second.any() and win_size > 0: # Ensure win_size is positive
        video_duration_seconds = len(audio_energy_per_second) * win_size
    if motion_score_per_second.any() and win_size > 0: # Ensure win_size is positive
        video_duration_seconds = max(video_duration_seconds, len(motion_score_per_second) * win_size)

    if video_duration_seconds == 0:
        if full_transcript:
            max_end_time = 0
            for seg_info in full_transcript: # Renamed 'seg' to 'seg_info' to avoid conflict
                if 'end_time' in seg_info and seg_info['end_time'] > max_end_time:
                    max_end_time = seg_info['end_time']
            if max_end_time > 0:
                video_duration_seconds = max_end_time
                log.info(f"Using video duration from transcript: {video_duration_seconds:.2f}s")

    if video_duration_seconds == 0:
         # Attempt to get duration from video file metadata as a last resort
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps > 0 and frame_count > 0:
                    video_duration_seconds = frame_count / fps
                    log.info(f"Using video duration from metadata: {video_duration_seconds:.2f}s")
            cap.release()
        except Exception as e_meta:
            log.warning(f"Could not get video duration from metadata: {e_meta}")

        if video_duration_seconds == 0: # If still zero
            log.error("Video duration could not be determined. Cannot select segments.")
            return []
    log.info(f"Determined video duration: {video_duration_seconds:.2f}s")

    # --- LLM-based candidate segment identification ---
    llm_suggested_segments = []
    if full_transcript and _llm and _tokenizer:
        log.info("🧠 Attempting to get candidate segments from LLM...")
        llm_suggested_segments = await get_llm_candidate_segments(full_transcript, video_duration_seconds) # Now awaited
        if llm_suggested_segments:
            log.info(f"✅ LLM returned {len(llm_suggested_segments)} candidate segments.")
        else:
            log.warning("ℹ️ LLM did not return any candidate segments. No clips will be processed.")
            return [] # If LLM provides no segments, we stop.
    elif not full_transcript:
        log.warning("ℹ️ Skipping LLM candidate segment identification (no transcript). No clips will be processed.")
        return []
    else: # LLM not loaded
        log.warning("ℹ️ LLM not loaded. Cannot identify segments. No clips will be processed.")
        return []

    potential_clips = []
    log.info(f"⚙️ Processing {len(llm_suggested_segments)} LLM-suggested segments...")

    for i, segment_suggestion in enumerate(llm_suggested_segments):
        start_time = segment_suggestion.get('start_time')
        end_time = segment_suggestion.get('end_time')

        if start_time is None or end_time is None:
            log.warning(f"Skipping LLM suggestion #{i+1} due to missing start/end times: {segment_suggestion}")
            continue

        # Validate and Adjust Segment Times
        start_time = float(start_time)
        end_time = float(end_time)

        if not (0 <= start_time < video_duration_seconds and 0 < end_time <= video_duration_seconds and start_time < end_time):
            log.warning(f"Skipping LLM suggestion #{i+1} due to invalid/out-of-bounds times: Start {start_time}, End {end_time}. Theme: {segment_suggestion.get('theme')}")
            continue

        original_duration = end_time - start_time
        clip_duration = original_duration

        # Simple adjustment logic
        adjusted = False
        if clip_duration < min_len:
            # Try to extend end_time
            potential_end_time = start_time + min_len
            if potential_end_time <= video_duration_seconds:
                end_time = potential_end_time
                clip_duration = end_time - start_time
                adjusted = True
                log.info(f"Segment #{i+1} too short ({original_duration:.2f}s). Adjusted end_time to meet MIN_CLIP_LEN ({min_len}s). New duration: {clip_duration:.2f}s")
            else: # Cannot extend enough
                log.warning(f"Skipping segment #{i+1} ('{segment_suggestion.get('theme')}'). Original duration {original_duration:.2f}s is less than MIN_CLIP_LEN ({min_len}s) and cannot be extended sufficiently.")
                continue
        elif clip_duration > max_len:
            # Try to shorten by reducing end_time
            end_time = start_time + max_len
            clip_duration = end_time - start_time # should be max_len
            adjusted = True
            log.info(f"Segment #{i+1} too long ({original_duration:.2f}s). Adjusted end_time to meet MAX_CLIP_LEN ({max_len}s). New duration: {clip_duration:.2f}s")

        if adjusted and (end_time > video_duration_seconds or start_time >= end_time): # Double check after adjustment
             log.warning(f"Skipping segment #{i+1} ('{segment_suggestion.get('theme')}') after adjustment led to invalid times. Start: {start_time}, End: {end_time}")
             continue


        timestamp_info = {'start': start_time, 'end': end_time}
        text_analysis_results = analyze_text(segment_suggestion, timestamp_info)
        visual_analysis_results = analyze_visuals(video_path, timestamp_info)

        start_idx = int(start_time / win_size) if win_size > 0 else 0
        end_idx = int(end_time / win_size) if win_size > 0 else 0
        
        avg_audio = 0
        if audio_energy_per_second.any() and start_idx < end_idx and end_idx <= len(audio_energy_per_second) and start_idx < len(audio_energy_per_second):
            valid_audio_slice = audio_energy_per_second[start_idx:end_idx]
            avg_audio = np.mean(valid_audio_slice) if valid_audio_slice.any() else 0
        elif not audio_energy_per_second.any():
            log.debug(f"No audio energy data for segment {i+1}")
        else:
            log.debug(f"Invalid audio slice indices for segment {i+1}: start_idx={start_idx}, end_idx={end_idx}, len={len(audio_energy_per_second)}")


        avg_motion = 0
        if motion_score_per_second.any() and start_idx < end_idx and end_idx <= len(motion_score_per_second) and start_idx < len(motion_score_per_second):
            valid_motion_slice = motion_score_per_second[start_idx:end_idx]
            avg_motion = np.mean(valid_motion_slice) if valid_motion_slice.any() else 0
        elif not motion_score_per_second.any():
            log.debug(f"No motion score data for segment {i+1}")
        else:
            log.debug(f"Invalid motion slice indices for segment {i+1}: start_idx={start_idx}, end_idx={end_idx}, len={len(motion_score_per_second)}")
            
        virality_score = calculate_virality_score(text_analysis_results, visual_analysis_results, avg_audio, avg_motion)
        log.info(f"  Segment #{i+1} ('{segment_suggestion.get('theme')}') [Dur: {clip_duration:.2f}s]: Virality Score = {virality_score:.2f}")
        potential_clips.append({
            'start': start_time,
            'end': end_time,
            'score': virality_score,
            'duration': clip_duration, # Use adjusted duration
            'llm_reason': segment_suggestion.get('reason_for_highlight', '')
        })

    if not potential_clips:
        log.warning("No potential clips generated after processing LLM suggestions.")
        return []

    potential_clips.sort(key=lambda x: x['score'], reverse=True)

    selected_clips = []
    chosen_intervals = []
    for clip_candidate in potential_clips:
        if len(selected_clips) >= nclips: # nclips is TARGET_CLIPS
            break

        cs = clip_candidate['start']
        ce = clip_candidate['end']

        is_overlapping = False
        for chosen_s, chosen_e in chosen_intervals:
            if cs < chosen_e and chosen_s < ce:
                is_overlapping = True
                break

        if not is_overlapping:
            selected_clips.append(clip_candidate)
            chosen_intervals.append((cs, ce))

    log.info(f"🏆 Selected {len(selected_clips)} non-overlapping clips from {len(potential_clips)} scored LLM suggestions.")

    if len(selected_clips) < nclips:
        log.warning(f"Fewer than {nclips} clips selected. Found {len(selected_clips)}.")
    # If len(selected_clips) > nclips, it's already handled by the loop break condition.

    return selected_clips

def cut_clips(video_path: str, segments: List[Dict[str, float]], outdir="clips"):
    os.makedirs(outdir, exist_ok=True); outputs = []
    for k, seg_info in enumerate(segments, 1):
        s = seg_info['start']; e = seg_info['end']; name = os.path.join(outdir, f"clip{k}_score{seg_info['score']:.2f}.mp4")
        cmd = ["ffmpeg", "-y", "-i", video_path, "-ss", str(s), "-to", str(e), "-vf", "format=yuv420p",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "128k",
               "-movflags", "+faststart", name]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            outputs.append(name)
        except subprocess.CalledProcessError as e_cut:
            log.error(f"ERROR cutting clip [cyan]{name}[/cyan]: {e_cut.stderr.decode('utf-8') if e_cut.stderr else e_cut}", exc_info=True)
    return outputs

async def main():
    """
    Main asynchronous function to run the video clip extraction pipeline.

    Parses command-line arguments, loads models, processes the video (input handling,
    transcription, audio/motion analysis), identifies best segments using an LLM-driven
    approach with multi-modal scoring, cuts the selected clips, and logs performance metrics
    including VRAM usage if a GPU is utilized.
    """
    parser = argparse.ArgumentParser(description="Extracts viral clips from videos.")
    parser.add_argument("video_input_source", type=str, help="URL or local path of the video to process.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("-o", "--output-dir", type=str, default="clips", help="Directory to save extracted clips.")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("rich").setLevel(logging.DEBUG)
        log.debug("🐞 Debug mode enabled.")

    log.info("🚀 Starting Clip Extractor Pipeline...")
    load_stt_model() 
    load_llm_model() 
    load_vision_models()

    if DEVICE.type == "cuda":
        log.info(f"VRAM after model loading: Allocated: {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB, Max Allocated: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB, Reserved: {torch.cuda.memory_reserved(DEVICE)/1e9:.2f} GB, Max Reserved: {torch.cuda.max_memory_reserved(DEVICE)/1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats(DEVICE) # Reset peak stats before main processing

    video_input_source = args.video_input_source
    output_clips_dir = args.output_dir

    processed_video_path = None 
    log.info(f"▶️ Processing input: [bold cyan]{video_input_source}[/bold cyan]")

    if validators.url(video_input_source):
        log.info("🌐 Input identified as URL. Attempting download...")
        download_output_dir = os.path.join(MODELS_DIR, "downloaded_videos") 
        os.makedirs(download_output_dir, exist_ok=True)
        downloaded_path = download_video(video_input_source, output_dir=download_output_dir)
        if downloaded_path: processed_video_path = downloaded_path; log.info(f"✅ Video downloaded: [green]{processed_video_path}[/green]")
        else: log.critical(f"❌ Failed to download video from URL: {video_input_source}. Exiting."); return
    else:
        log.info(f"📁 Input identified as local file path: [green]{video_input_source}[/green]")
        if os.path.exists(video_input_source): processed_video_path = video_input_source; log.info(f"✅ Local video found.")
        else: log.critical(f"❌ Local video file not found: {video_input_source}. Exiting."); return 
    
    if not processed_video_path: log.critical("❌ No valid video to process after input handling. Exiting."); return

    full_transcript = []
    stt_start_time = time.perf_counter()
    try:
        log.info("🎙️ Transcribing video...")
        full_transcript = transcribe_video(processed_video_path) # This is synchronous
        stt_duration = time.perf_counter() - stt_start_time
        log.info(f"✅ Transcription completed in {stt_duration:.2f} seconds. Segments: {len(full_transcript)}")
        if not full_transcript: log.warning("Transcription returned no segments.")
    except Exception as e: log.critical(f"ERROR during transcription: {e}\nCould not perform transcription. Exiting.", exc_info=True); return 

    audio_rms = np.array([])
    audio_rms_start_time = time.perf_counter()
    try: 
        log.info("🎧 Analyzing audio for energy peaks...")
        audio_rms = audio_peaks(processed_video_path, WIN_SIZE) # Synchronous
        audio_rms_duration = time.perf_counter() - audio_rms_start_time
        log.info(f"✅ Audio RMS analysis completed in {audio_rms_duration:.2f} seconds. RMS array shape: {audio_rms.shape}")
        if not audio_rms.any(): log.warning("Audio analysis (RMS) returned empty results.")
    except Exception as e: log.error(f"ERROR during audio analysis: {e}", exc_info=True); audio_rms = np.array([]) 

    motion_values = np.array([])
    motion_total_duration_log = 0.0
    motion_core_proc_time_log = 0.0
    motion_effective_fps_log = 0.0
    motion_start_time = time.perf_counter()
    try: 
        log.info("🖼️ Analyzing video for motion...")
        # motion_peaks now returns: motion_values_np, loop_duration_sec, total_frames_processed_in_loop
        motion_values, motion_core_proc_time, motion_frames_processed = motion_peaks(processed_video_path, WIN_SIZE) # Synchronous
        motion_total_duration = time.perf_counter() - motion_start_time
        motion_total_duration_log = motion_total_duration
        motion_core_proc_time_log = motion_core_proc_time
        log.info(f"✅ Motion analysis calculated in {motion_total_duration:.2f} seconds (core processing: {motion_core_proc_time:.2f}s). Motion array shape: {motion_values.shape}")
        if motion_core_proc_time > 0 and motion_frames_processed > 0:
            motion_effective_fps_log = motion_frames_processed / motion_core_proc_time
            log.info(f"Motion analysis effective FPS: {motion_effective_fps_log:.2f}")
        if not motion_values.any(): log.warning("Motion analysis returned empty results.")
    except Exception as e: log.error(f"ERROR during motion analysis: {e}", exc_info=True); motion_values = np.array([])

    segments = []
    segment_id_start_time = time.perf_counter()
    try:
        log.info("🧠 Identifying best segments with AI...")
        segments = await best_segments(processed_video_path, full_transcript, audio_rms, motion_values, WIN_SIZE, TARGET_CLIPS, MIN_CLIP_LEN, MAX_CLIP_LEN) # Awaited
        segment_id_duration = time.perf_counter() - segment_id_start_time
        log.info(f"✅ Segment identification and analysis (LLM, Visuals) completed in {segment_id_duration:.2f} seconds. Found {len(segments)} potential clips.")
    except Exception as e: log.error(f"ERROR during segment identification: {e}. Exiting.", exc_info=True); return
    
    if DEVICE.type == "cuda": # VRAM after main processing
        log.info(f"VRAM after main processing (best_segments): Allocated: {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB, Max Allocated: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB, Reserved: {torch.cuda.memory_reserved(DEVICE)/1e9:.2f} GB, Max Reserved: {torch.cuda.max_memory_reserved(DEVICE)/1e9:.2f} GB")

    files = []    
    cutting_start_time = time.perf_counter()
    log.info(f"🔖 Selected Segments (Top {TARGET_CLIPS}):")
    if segments:
        for i, seg_info in enumerate(segments):
            start_time_str = f"{seg_info['start']:.2f}"; end_time_str = f"{seg_info['end']:.2f}"; virality_score_str = f"{seg_info['score']:.2f}"
            log.info(f"  Clip #{i+1}: [{start_time_str}s - {end_time_str}s], Score: [bold yellow]{virality_score_str}[/bold yellow]")
        try:
            log.info(f"✂️  Cutting clips into '[bold green]{os.path.abspath(output_clips_dir)}[/bold green]/'...")
            files = cut_clips(processed_video_path, segments, outdir=output_clips_dir) # Synchronous
            cutting_duration = time.perf_counter() - cutting_start_time
            log.info(f"✅ Clip cutting completed in {cutting_duration:.2f} seconds.")
            if not files: log.warning("`cut_clips` executed but returned no file paths. Clips might be missing.")
            else: log.info(f"✅🎞️ Generated clip files: {[str(f) for f in files]}")
        except Exception as e: log.error(f"ERROR during clip cutting: {e}", exc_info=True)
    else:
        cutting_duration = time.perf_counter() - cutting_start_time # Still log duration even if no clips
        log.info(f"ℹ️ No segments were selected by `best_segments` to be cut. Cutting duration: {cutting_duration:.2f}s.")
    
    if not files: log.warning("🏁 Pipeline finished. ⚠️ No clips were ultimately generated or saved.")
    else: log.info(f"🏁 Pipeline finished successfully. {len(files)} clips generated in '[bold green]{os.path.abspath(output_clips_dir)}[/bold green]'.")

if __name__ == "__main__":
    asyncio.run(main())

# mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm #
# --------------------- CONCEPTUAL TESTING STRATEGY ----------------------- #
# This script, in its current state, uses placeholders for core AI functionalities.
# Testing will focus on data flow, logic with dummy data, and output format.
#
# 1. PREPARATION:
#    - Ensure all dependencies from requirements.txt are installed (especially AI libraries).
#    - Have `ffmpeg` installed and in PATH.
#    - Prepare a short test video (e.g., 1-5 minutes) or use a YouTube URL.
#      - `video_input_source` in `main()` can be set for this.
#    - Create a `models/` directory in the same location as this script if it doesn't exist,
#      especially if `MODEL_PATH_STT` points to a local path initially or for YOLOv5 weights.
#
# 2. UNIT TESTS (Conceptual - would be separate test files in practice):
#    - `load_stt_model`, `load_llm_model`, `load_vision_models`:
#      - Test if models load correctly (if libraries are installed).
#      - Check behavior when libraries are missing (graceful fallback to simulation).
#      - For `load_vision_models`, test YOLOv5 weights path handling (missing vs. present).
#    - `transcribe_video`:
#      - With STT model loaded: provide a short audio segment, check for non-empty transcript.
#      - With STT model NOT loaded: check for simulated output.
#    - `analyze_text` / `analyze_visuals`:
#      - With models loaded: check if they process input and return structured dicts (even if placeholder logic).
#      - With models NOT loaded: check for simulated outputs.
#    - `get_transcript_for_window`: (as before)
#    - `calculate_virality_score`: (as before)
#
# 3. INTEGRATION TEST (Running the main script):
#    - **Model Loading:**
#      - Do all `load_*_model` functions execute?
#      - Does the script correctly detect if libraries like `faster_whisper` or `transformers` are missing and print warnings?
#      - Does STT model download (if `openai/whisper-small` is used and not cached) or load from `MODEL_PATH_STT`?
#      - Does YOLOv5 attempt to load `yolov5s.pt` from `MODELS_DIR`? Does it warn if not found?
#    - **Video Input Handling:**
#      - Test with a valid YouTube URL: Does it download to `models/downloaded_videos`?
#      - Test with a valid local file path: Does it find and use it?
#      - Test with an invalid URL or non-existent local file: Does it exit gracefully?
#    - **Core Processing with Actual Models (if loaded):**
#      - `transcribe_video`: Is an actual transcript (not simulated) generated for the input video?
#      - `analyze_text` / `analyze_visuals`: Are they called and do they use the (partially implemented) models?
#        (Actual output quality is not the focus here, but data flow).
#    - **`best_segments` Logic & `cut_clips` Execution:** (as before, but now with potentially real STT data)
#      - Check segment count, duration, non-overlapping logic.
#      - Do clips get generated in `clips/` with scores in filenames?
#    - **Output Logging:**
#      - Are progress messages (loading, transcribing, analyzing, cutting) clear?
#      - Are errors and warnings informative?
#
# 4. CONFIGURATION TESTING: (as before)
#    - `TARGET_CLIPS`, `MIN_CLIP_LEN`, `MAX_CLIP_LEN`.
#    - `VIRALITY_CONFIG` weights.
#    - `MODEL_PATH_STT`, `MODEL_PATH_LLM` (e.g., try another small LLM if available).
#
# 5. VRAM/Performance (Manual Observation - Advanced):
#    - If models are large, monitor VRAM usage during `load_*_model` and processing.
#    - Observe overall execution time. (Not for optimization yet, but for baseline).
#
# Actual AI model accuracy and output quality testing is a separate, more involved process.
# mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm #