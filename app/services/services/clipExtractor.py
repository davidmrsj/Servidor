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
from app.services.services.download_youtube_video import download_video
import validators 
from transformers import BitsAndBytesConfig, pipeline

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
# Attempt to import AI models; handle ImportError if they are not installed
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None # type: ignore
    print("⚠️ WARNING: faster_whisper not installed. Transcription will be simulated if model cannot be loaded.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    AutoTokenizer = None # type: ignore
    AutoModelForCausalLM = None # type: ignore
    print("⚠️ WARNING: transformers not installed. LLM-based text analysis will be simulated.")

try:
    # Assuming yolov5 is installed from a package that makes it importable
    # This might vary depending on the specific YOLOv5 fork/package used.
    # Common ways: from yolov5 import YOLOv5, or torch.hub.load(...)
    # For this example, let's assume a hypothetical YOLOv5 class is available.
    # If using torch.hub.load, the loading logic would be different.
    class YOLOv5Placeholder: # Placeholder if actual import fails or is complex
        def __init__(self, weights, device): self.weights = weights; self.device = device; print(f"YOLOv5Placeholder initialized with weights: {weights}")
        def predict(self, frame): print("YOLOv5Placeholder: predict called"); return [] 
    YOLOv5 = YOLOv5Placeholder # Default to placeholder

    # Attempt to import a specific YOLOv5 package if available.
    # This is speculative and depends on what 'pip install yolov5' provides.
    # If a specific package is used, this import should match it.
    # For example, if it's 'yolov5torch': from yolov5torch import YOLOv5Model as YOLOv5
    # For now, we rely on the placeholder if no specific known 'yolov5' package is found.
    # If torch.hub.load is preferred, that logic would replace this.
    # Example of dynamic import attempt (adjust as needed for actual package):
    try:
        from yolov5 import YOLOv5 as ActualYOLOv5 # Try a common name
        YOLOv5 = ActualYOLOv5
        print("ℹ️  Successfully imported 'yolov5.YOLOv5'.")
    except ImportError:
        try:
            # Example for yolov5 from ultralytics hub
            import torch
            _yolo_model_hub_temp = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            # This is just to check if torch.hub works, the actual model usage is below
            print("ℹ️  Successfully checked torch.hub.load for ultralytics/yolov5.")
            # Note: The actual _yolo_model will be loaded in load_vision_models using a specific path
        except Exception as e_yolo_hub:
            print(f"⚠️ WARNING: Could not import 'yolov5' or load from torch.hub ({e_yolo_hub}). YOLOv5 detection will be simulated by placeholder.")

except ImportError:
    # This top-level except is for the initial "class YOLOv5Placeholder" block
    print("⚠️ WARNING: YOLOv5 related imports failed at placeholder definition. YOLOv5 detection will be simulated.")
    # YOLOv5 remains YOLOv5Placeholder

try:
    from fer import FER
except ImportError:
    FER = None # type: ignore
    print("⚠️ WARNING: fer (Facial Emotion Recognition) not installed. Emotion detection will be simulated.")


# Global variables for models (to be loaded once)
_stt_model = None
_llm = None
_tokenizer = None
_yolo_model = None
_fer_detector = None

# ───────── DIRECTORIES AND CONFIGS ───────── #
MODELS_DIR = "models" # For locally stored/downloaded models by this script
# Ensure this script is in a subdirectory of the main app for this path to work as intended
# Or use absolute paths / more robust path construction.
# For example: os.path.join(os.path.dirname(__file__), "models")

# ───────── CONFIGURACIÓN BÁSICA (Copied from previous state) ───────── #
TARGET_CLIPS = 10
WIN_SIZE     = 1.0
MIN_CLIP_LEN = 60
MAX_CLIP_LEN = 90
AUDIO_W      = 2.0
MOTION_W     = 1.0

# ───────── AI MODEL CONFIGURATION ───────── #
MODEL_PATH_STT = "openai/whisper-small" # Now directly Hugging Face model ID
MODEL_PATH_LLM = "meta-llama/Llama-2-7b-chat-hf"
MODEL_PATH_VISION_YOLO = "yolov5s.pt" # Will be joined with MODELS_DIR
# ────────────────────────────────────────── #

# ───────── VIRALITY SCORING CONFIGURATION (Copied from previous state) ───────── #
VIRALITY_CONFIG = {
    'weights': {
        'sentiment_positive': 1.5, 'sentiment_negative': 0.8, 'emotion_joy': 1.2,
        'emotion_surprise': 1.3, 'keyword_match': 2.0, 'engagement_hook': 1.5,
        'visual_intensity': 1.0, 'facial_expression_happy': 1.2, 'fast_cuts_or_action': 1.1,
        'audio_energy_avg': 0.5, 'motion_avg': 0.3,
    },
    'thresholds': {'min_sentiment_score': 0.5, 'min_visual_intensity': 0.2,},
    'trending_keywords': ['challenge', 'hack', 'reveal', 'shocking', 'must-see']
}
# ─────────────────────────────────────────────────── #

# ─── VRAM & EFFICIENCY STRATEGY (Copied from previous state) ─── #
# Given the 16GB VRAM constraint, running multiple large AI models simultaneously
# might be challenging. A sequential processing strategy could be:
# 1. Transcribe full video with STT (once).
# 2. For each candidate segment identified by `best_segments`:
#    a. Load/run Text Analysis (LLM). Unload LLM if VRAM is needed.
#    b. Load/run Visual Analysis (Vision Model). Unload Vision Model.
# This minimizes peak VRAM usage by not holding all models at once.
# The `analyze_text` and `analyze_visuals` functions would then handle
# their own model loading/unloading if this strategy is adopted,
# or models could be passed as arguments if managed centrally.
# ─────────────────────────────────────────────────── #

# ───────── MODEL LOADING FUNCTIONS ───────── #
def load_stt_model():
    global _stt_model
    if WhisperModel is None:
        print("ℹ️ faster_whisper library not available. STT model cannot be loaded. Transcription will be simulated.")
        return

    if _stt_model is None:
        try:
            print(f"🧠 Loading STT model: {MODEL_PATH_STT} (will download if not cached)...")
            # Model will be downloaded from Hugging Face if not cached.
            _stt_model = WhisperModel(
                model_size_or_path="openai/whisper-small", # Using direct model ID
                device="cuda",
                compute_type="float16",
                local_files_only=False # Allow model download if not cached
            )
            print("✅ STT Model loaded successfully.")
        except Exception as e:
            print(f"❌ ERROR loading STT model: {e}")
            print("Transcription will be simulated.")
            _stt_model = None

def load_llm_model():
    global _llm, _tokenizer # Global model variables
    
    if 'AutoTokenizer' not in globals() or AutoTokenizer is None or \
       'AutoModelForCausalLM' not in globals() or AutoModelForCausalLM is None or \
       'BitsAndBytesConfig' not in globals() or BitsAndBytesConfig is None:
        print("ℹ️ Key transformers components not available. LLM cannot be loaded.")
        return

    if _llm is None or _tokenizer is None: # Only load if not already loaded
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            print(f"🧠 Loading LLM model and tokenizer: {MODEL_PATH_LLM} (Llama-2, 4-bit)...")
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_LLM, cache_dir=MODELS_DIR)
            _llm = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH_LLM,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=MODELS_DIR,
                trust_remote_code=True
            )
            print(f"✅ LLM model ({MODEL_PATH_LLM}) and tokenizer loaded successfully.")
        except Exception as e:
            print(f"❌ ERROR loading LLM model ({MODEL_PATH_LLM}): {e}")
            print("    Ensure you have accepted Llama-2's license on Hugging Face.")
            print("    Ensure 'bitsandbytes' and 'accelerate' are installed.")
            _llm, _tokenizer = None, None

def load_vision_models():
    global _yolo_model, _fer_detector
    os.makedirs(MODELS_DIR, exist_ok=True) # Ensure models directory exists

    # Load YOLOv5
    if _yolo_model is None:
        try:
            print(f"🧠 Loading YOLOv5 model...")
            # For YOLOv5:
            # This setup expects 'yolov5s.pt' to be manually placed in MODELS_DIR.
            # Some yolov5 pip packages might download if 'yolov5s' (standard name) is given.
            # Check your specific yolov5 library's behavior for auto-downloading.
            yolo_weights_path = os.path.join(MODELS_DIR, 'yolov5s.pt') 
            
            if not os.path.exists(yolo_weights_path):
                print(f"⚠️ WARNING: YOLOv5 weights file '{yolo_weights_path}' not found.")
                print(f"Attempting to load 'yolov5s' (may trigger download by torch.hub if not cached)...")
                # This relies on torch.hub's caching mechanism.
                # The 'yolov5s' model is a standard small model.
                _yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
                # If specific weights are required, they must be placed manually,
                # or a direct download link + subprocess call could be added here.
            else:
                 _yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights_path, trust_repo=True)

            if hasattr(_yolo_model, 'to'): _yolo_model.to('cuda') # Move to GPU if applicable
            print("✅ YOLOv5 Model loaded successfully.")
        except Exception as e:
            print(f"❌ ERROR loading YOLOv5 model: {e}")
            print("YOLOv5 detection will be simulated by placeholder.")
            _yolo_model = YOLOv5Placeholder(weights="N/A", device="cpu") # Fallback to placeholder

    # Load FER
    if FER is not None and _fer_detector is None:
        try:
            print(f"🧠 Loading FER model (MTCNN and emotion classifier)...")
            # For FER:
            # The FER library typically auto-downloads required models (MTCNN, emotion classifier) on first use.
            _fer_detector = FER(mtcnn=True) # mtcnn=True for more accurate face detection
            print("✅ FER Model loaded successfully.")
        except Exception as e:
            print(f"❌ ERROR loading FER model: {e}")
            print("Facial emotion recognition will be simulated.")
            _fer_detector = None
# ─────────────────────────────────────────── #

def audio_peaks(video_path: str, win: float) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "44100", wav_path],
            capture_output=True, check=True,
        )
        y, sr = librosa.load(wav_path, sr=None)
        hop = int(sr * win)
        rms = librosa.feature.rms(y=y, frame_length=hop, hop_length=hop, center=False)[0]
    finally:
        if os.path.exists(wav_path): os.remove(wav_path)
    return rms

def motion_peaks(video_path: str, win: float) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: print("⚠️ Warning: Could not get FPS from video. Motion analysis might be incorrect."); return np.array([])
    step = int(fps * win)
    if step == 0: step = int(fps) # Default to 1s window if win_size is too small for FPS
    
    prev = None; mvals = []; idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))
            if prev is not None:
                diff = cv2.absdiff(gray, prev)
                mvals.append(diff.sum())
            prev = gray
        idx += 1
    cap.release()
    return np.array(mvals)

def transcribe_video(video_path: str) -> List[Dict]:
    if _stt_model is None:
        print(f"ℹ️  STT Model not loaded or failed to load. Simulating transcription for {video_path}")
        return [{'start_time': 0.0, 'end_time': 5.0, 'text': 'Simulated: STT model not available.'}]

    print(f"🎙️  Actual STT: Transcribing {video_path}...")
    audio_output_path = "" 
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            audio_output_path = tmp_audio.name
        
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", audio_output_path],
            capture_output=True, check=True, timeout=180 # 3 min timeout for audio extraction
        )

        segments_generator, info = _stt_model.transcribe(audio_output_path, beam_size=5)
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        transcript_data = []
        for segment in segments_generator:
            transcript_data.append({
                "start_time": round(segment.start, 2),
                "end_time": round(segment.end, 2),
                "text": segment.text.strip()
            })
        
        print("✅ Actual STT: Transcription complete.")
        return transcript_data
    except subprocess.TimeoutExpired:
        print(f"❌ ERROR: Timeout during ffmpeg audio extraction for STT from {video_path}.")
        print("Falling back to simulated transcription.")
        return [{'start_time': 0.0, 'end_time': 5.0, 'text': 'Simulated: Timeout during audio extraction for STT.'}]
    except Exception as e:
        print(f"❌ ERROR during actual STT transcription: {e}")
        print("Falling back to simulated transcription.")
        return [{'start_time': 0.0, 'end_time': 5.0, 'text': 'Simulated: Error during STT.'}]
    finally:
        if audio_output_path and os.path.exists(audio_output_path):
            try: os.remove(audio_output_path)
            except Exception as e_rem: print(f"Error removing temp audio file {audio_output_path}: {e_rem}")

def analyze_text(transcript_segment: str, timestamp_info: Dict) -> Dict:
    # TODO: Implement robust parsing of LLM output for structured data.
    if _llm is None or _tokenizer is None:
        print("ℹ️ LLM model/tokenizer not loaded. analyze_text will use placeholders.")
        return { 
            'timestamp_info': timestamp_info, 'sentiment': {'label': 'neutral_simulated', 'score': 0.5},
            'emotions': ['simulated_emotion'], 'keywords': ['simulated', 'keywords'],
            'topics': ['simulated_topic'], 'humor_detected': False, 'controversy_detected': False,
            'engagement_hooks': [], 'summary': "Simulated summary: LLM not available."
        }
    
    try:
        llm_pipeline_obj = pipeline(
            "text-generation", 
            model=_llm, 
            tokenizer=_tokenizer, 
            device_map=_llm.hf_device_map if hasattr(_llm, 'hf_device_map') else 'auto'
        )
        prompt = f"Analyze feelings, emotions, and keywords in the text: <<{transcript_segment}>>. Output labels: sentiment, emotions list, keywords list."
        generated_outputs = llm_pipeline_obj(prompt, max_new_tokens=150, num_return_sequences=1, do_sample=False)
        llm_raw_output = generated_outputs[0]['generated_text']
        
        return {
            'timestamp_info': timestamp_info, 'llm_output': llm_raw_output, 
            'sentiment': {'label': 'llm_needs_parsing', 'score': 0.5}, 
            'emotions': ['llm_needs_parsing'], 'keywords': ['llm_needs_parsing'],
            'topics': ['llm_needs_parsing'], 'summary': "LLM analysis (parsing needed)."
        }
    except Exception as e:
        print(f"❌ ERROR during LLM text analysis pipeline: {e}")
        return { 
            'timestamp_info': timestamp_info, 'sentiment': {'label': 'error_in_llm_exec', 'score': 0.0},
            'summary': f"Error in LLM exec: {e}"
        }
    
def analyze_visuals(video_segment_path: str, timestamp_info: Dict) -> Dict:
    # video_segment_path is currently a dummy string like "dummy_path_for_segment_X_Y.mp4"
    # In a real scenario, this would be an actual path to a subclip or frames.
    
    yolo_results_placeholder = []
    fer_results_placeholder = []

    if _yolo_model is not None and not isinstance(_yolo_model, YOLOv5Placeholder):
        # Placeholder: actual YOLO processing would happen on frames from video_segment_path
        # print(f"👁️ Actual YOLO: Analyzing visuals in {video_segment_path} (placeholder call)")
        yolo_results_placeholder = [{'object': 'person_yolo_placeholder', 'confidence': 0.8, 'bbox': [0,0,0,0]}]
    
    if _fer_detector is not None:
        # Placeholder: actual FER processing would happen on frames with detected faces
        # print(f"😊 Actual FER: Analyzing emotions in {video_segment_path} (placeholder call)")
        fer_results_placeholder = [{'box': [0,0,0,0], 'emotions': {'happy_fer_placeholder': 0.7, 'neutral': 0.3}}]

    return {
        'timestamp_info': timestamp_info,
        'facial_expressions': fer_results_placeholder if _fer_detector else [{'expression': 'simulated_neutral', 'confidence': 0.5}],
        'detected_actions': [], 
        'key_objects': yolo_results_placeholder if _yolo_model and not isinstance(_yolo_model, YOLOv5Placeholder) else [{'object': 'simulated_object'}],
        'scene_changes': [{'type': 'simulated_cut', 'timestamp': timestamp_info['start'] + 1.0}],
        'visual_intensity_score': 0.4, # Slightly different from pure dummy
        'notes': f"Visual analysis (actual models placeholder) for segment."
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
        # Simplified example for facial expression scoring from FER results
        for face in visual_analysis.get('facial_expressions', []):
            if isinstance(face, dict) and 'emotions' in face: # Check if it's a FER-like dict
                if face['emotions'].get('happy_fer_placeholder', 0.0) > 0.5: # Example check
                     score += weights.get('facial_expression_happy', 1.0)
            elif isinstance(face, dict) and face.get('expression') == 'happy': # For older placeholder
                 score += weights.get('facial_expression_happy', 1.0)

        if len(visual_analysis.get('scene_changes', [])) > 2: score += weights.get('fast_cuts_or_action', 1.0)
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
    if not audio_energy_per_second.any() : print("⚠️ Warning: Audio energy data is empty. Cannot select segments based on audio."); # Allow proceeding if motion is present
    if not motion_score_per_second.any(): print("⚠️ Warning: Motion score data is empty. Cannot select segments based on motion."); # Allow proceeding if audio is present
    if not audio_energy_per_second.any() and not motion_score_per_second.any(): print("❌ ERROR: Both audio and motion data are empty. Cannot select segments."); return []
    
    video_duration_seconds = 0
    if audio_energy_per_second.any():
        video_duration_seconds = len(audio_energy_per_second) * win_size
    elif motion_score_per_second.any(): # Fallback if audio_energy is empty but motion is not
        video_duration_seconds = len(motion_score_per_second) * win_size

    if video_duration_seconds == 0: print("⚠️ Warning: Video duration is zero based on available data. Cannot select segments."); return []

    analysis_window_duration = (min_len + max_len) // 2
    if analysis_window_duration <= 0: analysis_window_duration = 60; print(f"⚠️ Warning: Calculated analysis_window_duration was <=0. Defaulted to {analysis_window_duration}s.")
    step_size = max(15, analysis_window_duration // 4); potential_clips = []
    print(f"⚙️  Analyzing video of {video_duration_seconds:.2f}s. Window: {analysis_window_duration}s, Step: {step_size}s")

    for current_start_time in np.arange(0, video_duration_seconds - analysis_window_duration + 1, step_size):
        current_end_time = current_start_time + analysis_window_duration
        if current_end_time > video_duration_seconds: current_end_time = video_duration_seconds
        if current_start_time >= current_end_time : continue
        
        transcript_for_window = get_transcript_for_window(full_transcript, current_start_time, current_end_time)
        timestamp_info = {'start': current_start_time, 'end': current_end_time}
        text_analysis_results = analyze_text(transcript_for_window, timestamp_info)
        
        # For visual analysis, it's more complex. Ideally, we'd pass frames or the subclip.
        # For now, video_path (full video) is passed, but analyze_visuals should handle frame extraction for the window.
        visual_analysis_results = analyze_visuals(video_path, timestamp_info) # Pass full video path

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

    if not potential_clips: print("⚠️ No potential clips generated after analysis."); return []
    potential_clips.sort(key=lambda x: x['score'], reverse=True); selected_clips = []; chosen_intervals = []
    for clip_candidate in potential_clips:
        if len(selected_clips) >= nclips: break
        cs = clip_candidate['start']; ce = clip_candidate['end']
        is_overlapping = False
        for chosen_s, chosen_e in chosen_intervals:
            if cs < chosen_e and chosen_s < ce: is_overlapping = True; break
        if not is_overlapping: selected_clips.append({'start': cs, 'end': ce, 'score': clip_candidate['score']}); chosen_intervals.append((cs, ce))
    print(f"🏆 Selected {len(selected_clips)} clips out of {len(potential_clips)} potential clips.")
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
            print(f"❌ ERROR cutting clip {name}: {e_cut.stderr.decode('utf-8') if e_cut.stderr else e_cut}")
    return outputs

def main():
    """
    Main function to orchestrate the video processing pipeline.
    Handles input (local file or YouTube URL), downloads if necessary,
    then runs transcription, audio/visual analysis, segment selection,
    and clip cutting.
    """
    print("🚀 Starting Clip Extractor Pipeline...")
    load_stt_model() 
    load_llm_model() 
    load_vision_models()

    video_input_source = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Default to example
    # Example for local file: video_input_source = "input.mp4" 
    # video_input_source = os.getenv("VIDEO_INPUT", video_input_source) # Allow override from ENV

    video_path_for_processing = None
    print(f"▶️ Processing input: {video_input_source}")

    if validators.url(video_input_source):
        print(f"🌐 Input identified as URL. Attempting download...")
        download_output_dir = os.path.join(MODELS_DIR, "downloaded_videos") # Store in models/downloaded
        os.makedirs(download_output_dir, exist_ok=True)
        downloaded_path = download_video(video_input_source, output_dir=download_output_dir)
        if downloaded_path: video_path_for_processing = downloaded_path; print(f"✅ Video downloaded: {video_path_for_processing}")
        else: print(f"❌ Failed to download video from URL: {video_input_source}. Exiting."); return
    else:
        print(f"📁 Input identified as local file path: {video_input_source}")
        if os.path.exists(video_input_source): video_path_for_processing = video_input_source; print(f"✅ Local video found: {video_path_for_processing}")
        else: print(f"❌ Local video file not found: {video_input_source}. Exiting."); return
    
    if not video_path_for_processing: print("❌ No valid video to process after input handling. Exiting."); return

    full_transcript = []
    try:
        print("🎙️ Transcribing video...")
        full_transcript = transcribe_video(video_path_for_processing) 
        print(f"✅ Transcription complete. Segments: {len(full_transcript)}")
        if not full_transcript: print("⚠️ Warning: Transcription returned no segments.")
    except Exception as e: print(f"❌ ERROR during transcription: {e}\nCould not perform transcription. Exiting."); return

    audio_rms = np.array([])
    try: 
        print("🎧 Analyzing audio for energy peaks...")
        audio_rms = audio_peaks(video_path_for_processing, WIN_SIZE)
        print(f"✅ Audio analysis complete. RMS array shape: {audio_rms.shape}")
        if not audio_rms.any(): print("⚠️ Warning: Audio analysis (RMS) returned empty results.")
    except Exception as e: print(f"❌ ERROR during audio analysis: {e}\nCould not perform audio analysis. Proceeding without audio energy data if possible."); audio_rms = np.array([]) # Allow continuation

    motion = np.array([])
    try: 
        print("🖼️ Analyzing video for motion...")
        motion = motion_peaks(video_path_for_processing, WIN_SIZE)
        print(f"✅ Motion analysis complete. Motion array shape: {motion.shape}")
        if not motion.any(): print("⚠️ Warning: Motion analysis returned empty results.")
    except Exception as e: print(f"❌ ERROR during motion analysis: {e}\nCould not perform motion analysis. Proceeding without motion data if possible."); motion = np.array([]) # Allow continuation

    segments = []
    try:
        print("🧠 Identifying best segments with AI...")
        segments = best_segments(video_path_for_processing, full_transcript, audio_rms, motion, WIN_SIZE, TARGET_CLIPS, MIN_CLIP_LEN, MAX_CLIP_LEN)
        print(f"✅ Segment identification complete. Found {len(segments)} potential clips.")
    except Exception as e: print(f"❌ ERROR during segment identification: {e}\nCould not identify best segments. Exiting."); return
    
    files = []
    output_clips_dir = "clips" # Define output directory for generated clips
    print(f"🔖 Selected Segments (Top {TARGET_CLIPS}):")
    if segments:
        for i, seg_info in enumerate(segments):
            start_time_str = f"{seg_info['start']:.2f}"; end_time_str = f"{seg_info['end']:.2f}"; virality_score_str = f"{seg_info['score']:.2f}"
            print(f"  Clip #{i+1}: [{start_time_str}s - {end_time_str}s], Virality Score: {virality_score_str}")
        try:
            print(f"✂️  Cutting clips into '{output_clips_dir}/'...")
            files = cut_clips(video_path_for_processing, segments, outdir=output_clips_dir) 
            if not files: print("⚠️ WARNING: `cut_clips` executed but returned no file paths. Clips might be missing.")
            else: print(f"✅🎞️ Generated clip files: {files}")
        except Exception as e: print(f"❌ ERROR during clip cutting: {e}\nCould not cut clips.")
    else: print("ℹ️ No segments were selected by `best_segments` to be cut.")
    
    if not files: print("🏁 Pipeline finished. ⚠️ No clips were ultimately generated or saved.")
    else: print(f"🏁 Pipeline finished successfully. {len(files)} clips generated in '{os.path.abspath(output_clips_dir)}'.")

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