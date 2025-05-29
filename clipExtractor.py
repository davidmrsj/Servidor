"""
highlight_extractor.py
Identifies and extracts potentially viral short clips from a longer video.

Core Functionality:
- Handles video input (local file or YouTube URL, with download capability).
- Transcribes video audio using Whisper.
- Performs placeholder text analysis (LLM) and visual analysis (YOLO, FER).
- Calculates a 'virality score' for segments based on multi-modal analysis.
- Selects top N non-overlapping segments.
- Cuts selected segments using FFmpeg.

Key Dependencies:
  numpy, librosa, opencv-python, moviepy, yt-dlp, validators, ffmpeg (system path),
  faster_whisper, transformers, torch, yolov5, fer, bitsandbytes, accelerate.

Usage:
  Ensure all dependencies from requirements.txt are installed.
  Ensure 'ffmpeg' is accessible in the system PATH.
  Modify `video_input_source` in `main()` to a local path or YouTube URL.
  Run directly: `python clipExtractor.py`
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
import logging
import argparse
from rich.console import Console
from rich.logging import RichHandler

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
MIN_CLIP_LEN = 10 
MAX_CLIP_LEN = 35 
AUDIO_W      = 2.0
MOTION_W     = 1.0

# ───────── AI MODEL CONFIGURATION ───────── #
MODEL_PATH_STT = "small"
LLM_MODEL_PRIMARY = "meta-llama/Llama-2-7b-chat-hf"
LLM_MODEL_FALLBACK_OPEN = "mistralai/Mistral-7B-Instruct-v0.2" 
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

def motion_peaks(video_path: str, win: float) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Failed to open video: [cyan]{video_path}[/cyan] in motion_peaks.")
        return np.array([])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        log.warning(f"Could not get FPS from video [cyan]{video_path}[/cyan]. Motion analysis might be incorrect.")
        cap.release()
        return np.array([])
    step = int(fps * win)
    if step == 0:
        step = int(fps)
    
    prev_tensor = None
    mvals = []
    idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))
            gray_tensor = torch.from_numpy(gray).float().to(device)
            
            if prev_tensor is not None:
                diff = torch.abs(gray_tensor - prev_tensor)
                mvals.append(diff.sum().item())
            
            prev_tensor = gray_tensor
        idx += 1
    cap.release()
    return np.array(mvals)

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
        segments_gen, info = _stt_model.transcribe(audio_output_path, beam_size=5)
        log.info(
            f"Detected language '[italic yellow]{info.language}[/italic yellow]' "
            f"prob={info.language_probability:.2f}"
        )

        transcript = [
            {
                "start_time": round(seg.start, 2),
                "end_time": round(seg.end, 2),
                "text": seg.text.strip(),
            }
            for seg in segments_gen
        ]

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


def analyze_text(transcript_segment: str, timestamp_info: Dict) -> Dict:
    default_response = {
        'timestamp_info': timestamp_info,
        'sentiment': {'label': 'neutral_simulated', 'score': 0.5},
        'emotions': ['simulated_emotion'],
        'keywords': ['simulated', 'keywords'],
        'raw_llm_output': '',
        'summary': "Simulated summary: LLM not available or error during processing."
    }

    if _llm is None or _tokenizer is None:
        log.warning("LLM model/tokenizer not loaded for text analysis. Using placeholders.")
        return default_response

    prompt = f"""Analyze the following text and provide:
1. Sentiment (positive, negative, neutral) and a confidence score (0.0-1.0).
2. A list of up to 3 dominant emotions (e.g., joy, sadness, anger, surprise).
3. A list of up to 5 relevant keywords.

Format your response as a JSON object like this:
{{
  "sentiment": {{"label": "positive", "score": 0.9}},
  "emotions": ["joy", "excitement"],
  "keywords": ["event", "announcement", "new"]
}}

Text: <<{transcript_segment}>>
"""

    llm_raw_output = ""
    try:
        # ── determinar dispositivo para el pipeline ──
        pipeline_device = -1
        if hasattr(_llm, "device") and _llm.device is not None:
            if _llm.device.type == "cuda":
                pipeline_device = _llm.device.index if _llm.device.index is not None else 0
        elif DEVICE.type == "cuda":
            pipeline_device = 0

        text_gen_pipeline = pipeline(
            "text-generation",
            model=_llm,
            tokenizer=_tokenizer,
        )

        outputs = text_gen_pipeline(
            prompt,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.0,
            top_k=1
        )

        if outputs and isinstance(outputs, list) and "generated_text" in outputs[0]:
            llm_raw_output = outputs[0]["generated_text"]
            if prompt in llm_raw_output:
                llm_raw_output = llm_raw_output.split(prompt)[-1].strip()

            try:
                json_match = None
                if "```json" in llm_raw_output:
                    json_str = llm_raw_output.split("```json")[-1].split("```")[0].strip()
                    json_match = json.loads(json_str)
                elif "{" in llm_raw_output and "}" in llm_raw_output:
                    start_index = llm_raw_output.find("{")
                    end_index = llm_raw_output.rfind("}") + 1
                    if start_index != -1 and end_index != -1:
                        json_str = llm_raw_output[start_index:end_index]
                        json_match = json.loads(json_str)

                if json_match and isinstance(json_match, dict):
                    sentiment = json_match.get("sentiment", default_response["sentiment"])
                    emotions = json_match.get("emotions", default_response["emotions"])
                    keywords = json_match.get("keywords", default_response["keywords"])

                    if not (isinstance(sentiment, dict) and "label" in sentiment and "score" in sentiment):
                        sentiment = default_response["sentiment"]
                    if not isinstance(emotions, list):
                        emotions = default_response["emotions"]
                    if not isinstance(keywords, list):
                        keywords = default_response["keywords"]

                    return {
                        "timestamp_info": timestamp_info,
                        "sentiment": sentiment,
                        "emotions": emotions,
                        "keywords": keywords,
                        "raw_llm_output": llm_raw_output,
                        "summary": "LLM analysis complete (parsed successfully).",
                    }
                else:
                    raise ValueError("Parsed JSON is not a valid dictionary or not found.")

            except Exception as e_parse:
                log.warning(f"Could not parse JSON from LLM output: {e_parse}", exc_info=False)
                log.debug(f"   LLM Raw Output for parsing failure: {llm_raw_output}")
                return {
                    "timestamp_info": timestamp_info,
                    "sentiment": default_response["sentiment"],
                    "emotions": default_response["emotions"],
                    "keywords": default_response["keywords"],
                    "raw_llm_output": llm_raw_output,
                    "summary": f"LLM analysis complete (JSON parsing failed: {e_parse}).",
                }
        else:
            llm_raw_output = "No output from LLM pipeline."
            raise ValueError(llm_raw_output)

    except Exception as e_pipeline:
        log.error(f"ERROR during LLM text analysis pipeline: {e_pipeline}", exc_info=True)
        return {
            "timestamp_info": timestamp_info,
            "sentiment": default_response["sentiment"],
            "emotions": default_response["emotions"],
            "keywords": default_response["keywords"],
            "raw_llm_output": llm_raw_output,
            "summary": f"Error in LLM execution: {e_pipeline}",
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

def best_segments(
    video_path: str, full_transcript: List[Dict],
    audio_energy_per_second: np.ndarray, motion_score_per_second: np.ndarray,
    win_size: float, nclips: int = TARGET_CLIPS,
    min_len: int = MIN_CLIP_LEN, max_len: int = MAX_CLIP_LEN
) -> List[Dict]:
    if not audio_energy_per_second.any() : log.warning("Audio energy data is empty for segment scoring.");
    if not motion_score_per_second.any(): log.warning("Motion score data is empty for segment scoring.");
    if not audio_energy_per_second.any() and not motion_score_per_second.any(): 
        log.error("Both audio and motion data are empty. Cannot select segments based on these criteria.")
        return [] 
    
    video_duration_seconds = 0
    if audio_energy_per_second.any():
        video_duration_seconds = len(audio_energy_per_second) * win_size
    elif motion_score_per_second.any(): 
        video_duration_seconds = len(motion_score_per_second) * win_size

    if video_duration_seconds == 0: 
        log.warning("Video duration is zero based on available audio/motion data. Cannot select segments.")
        return []

    analysis_window_duration = (min_len + max_len) // 2 
    if analysis_window_duration <= 0: 
        log.warning(f"Calculated analysis_window_duration was <=0. Defaulting to {(MIN_CLIP_LEN + MAX_CLIP_LEN) // 2}s.")
        analysis_window_duration = (MIN_CLIP_LEN + MAX_CLIP_LEN) // 2
    
    step_size = 5 
    
    potential_clips = []
    log.info(f"⚙️  Analyzing video of {video_duration_seconds:.2f}s. Target segment length: ~{analysis_window_duration}s, Step: {step_size}s")

    for current_start_time in np.arange(0, video_duration_seconds - analysis_window_duration + 1e-9, step_size): 
        current_end_time = current_start_time + analysis_window_duration
        if current_end_time > video_duration_seconds: current_end_time = video_duration_seconds
        if current_start_time >= current_end_time : continue
        
        transcript_for_window = get_transcript_for_window(full_transcript, current_start_time, current_end_time)
        timestamp_info = {'start': current_start_time, 'end': current_end_time}
        text_analysis_results = analyze_text(transcript_for_window, timestamp_info)
        
        visual_analysis_results = analyze_visuals(video_path, timestamp_info) 

        start_idx = int(current_start_time / win_size); end_idx = int(current_end_time / win_size)
        
        avg_audio = 0
        if audio_energy_per_second.any():
            valid_audio_slice = audio_energy_per_second[start_idx:end_idx] if start_idx < end_idx and start_idx < len(audio_energy_per_second) else np.array([0])
            avg_audio = np.mean(valid_audio_slice) if valid_audio_slice.any() else 0
        
        avg_motion = 0
        if motion_score_per_second.any():
            valid_motion_slice = motion_score_per_second[start_idx:end_idx] if start_idx < end_idx and start_idx < len(motion_score_per_second) else np.array([0])
            avg_motion = np.mean(valid_motion_slice) if valid_motion_slice.any() else 0
            
        virality_score = calculate_virality_score(text_analysis_results, visual_analysis_results, avg_audio, avg_motion)
        potential_clips.append({'start': current_start_time, 'end': current_end_time, 'score': virality_score, 'duration': current_end_time - current_start_time})

    if not potential_clips: log.warning("No potential clips generated after analysis."); return []
    potential_clips.sort(key=lambda x: x['score'], reverse=True); selected_clips = []; chosen_intervals = []
    for clip_candidate in potential_clips:
        if len(selected_clips) >= nclips: break
        cs = clip_candidate['start']; ce = clip_candidate['end']
        is_overlapping = False
        for chosen_s, chosen_e in chosen_intervals:
            if cs < chosen_e and chosen_s < ce: is_overlapping = True; break
        if not is_overlapping: selected_clips.append({'start': cs, 'end': ce, 'score': clip_candidate['score']}); chosen_intervals.append((cs, ce))
    log.info(f"🏆 Selected {len(selected_clips)} clips out of {len(potential_clips)} potential clips.")
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

def main():
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
    try:
        log.info("🎙️ Transcribing video...")
        full_transcript = transcribe_video(processed_video_path) 
        log.info(f"✅ Transcription complete. Segments: {len(full_transcript)}")
        if not full_transcript: log.warning("Transcription returned no segments.")
    except Exception as e: log.critical(f"ERROR during transcription: {e}\nCould not perform transcription. Exiting.", exc_info=True); return 
    audio_rms = np.array([])
    try: 
        log.info("🎧 Analyzing audio for energy peaks...")
        audio_rms = audio_peaks(processed_video_path, WIN_SIZE)
        log.info(f"✅ Audio analysis complete. RMS array shape: {audio_rms.shape}")
        if not audio_rms.any(): log.warning("Audio analysis (RMS) returned empty results.")
    except Exception as e: log.error(f"ERROR during audio analysis: {e}", exc_info=True); audio_rms = np.array([]) 
    motion = np.array([])
    try: 
        log.info("🖼️ Analyzing video for motion...")
        motion = motion_peaks(processed_video_path, WIN_SIZE)
        log.info(f"✅ Motion analysis complete. Motion array shape: {motion.shape}")
        if not motion.any(): log.warning("Motion analysis returned empty results.")
    except Exception as e: log.error(f"ERROR during motion analysis: {e}", exc_info=True); motion = np.array([]) 
    segments = []
    try:
        log.info("🧠 Identifying best segments with AI...")
        segments = best_segments(processed_video_path, full_transcript, audio_rms, motion, WIN_SIZE, TARGET_CLIPS, MIN_CLIP_LEN, MAX_CLIP_LEN)
        log.info(f"✅ Segment identification complete. Found {len(segments)} potential clips.")
    except Exception as e: log.error(f"ERROR during segment identification: {e}. Exiting.", exc_info=True); return
    
    files = []    
    log.info(f"🔖 Selected Segments (Top {TARGET_CLIPS}):")
    if segments:
        for i, seg_info in enumerate(segments):
            start_time_str = f"{seg_info['start']:.2f}"; end_time_str = f"{seg_info['end']:.2f}"; virality_score_str = f"{seg_info['score']:.2f}"
            log.info(f"  Clip #{i+1}: [{start_time_str}s - {end_time_str}s], Score: [bold yellow]{virality_score_str}[/bold yellow]")
        try:
            log.info(f"✂️  Cutting clips into '[bold green]{os.path.abspath(output_clips_dir)}[/bold green]/'...")
            files = cut_clips(processed_video_path, segments, outdir=output_clips_dir) 
            if not files: log.warning("`cut_clips` executed but returned no file paths. Clips might be missing.")
            else: log.info(f"✅🎞️ Generated clip files: {[str(f) for f in files]}")
        except Exception as e: log.error(f"ERROR during clip cutting: {e}", exc_info=True)
    else: log.info("ℹ️ No segments were selected by `best_segments` to be cut.")
    
    if not files: log.warning("🏁 Pipeline finished. ⚠️ No clips were ultimately generated or saved.")
    else: log.info(f"🏁 Pipeline finished successfully. {len(files)} clips generated in '[bold green]{os.path.abspath(output_clips_dir)}[/bold green]'.")

if __name__ == "__main__":
    main()

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