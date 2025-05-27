"""
highlight_extractor.py
Identifies and extracts potentially viral short clips from a longer video.

Core Functionality:
- Analyzes video for audio peaks (RMS energy) and visual motion (frame differences).
- (Placeholder) Transcribes audio and performs AI-based text and visual analysis on
  sliding windows of the video to calculate a 'virality score'.
- Selects the top N non-overlapping segments based on these scores.
- Uses FFmpeg to cut the selected segments into individual clip files.

Input:
- Can process a local video file or download a video from a YouTube URL.

Output:
- Generates .mp4 clip files in a specified output directory (default 'clips/').
- Prints progress, selected segment info, and error messages to the console.

Key Dependencies:
  numpy, librosa, opencv-python, moviepy, yt-dlp, validators, ffmpeg (system path),
  faster_whisper, transformers (for future LLM/Vision).

Usage:
  Modify `video_input_source` in `main()` to a local path or YouTube URL.
  Run directly: `python clipExtractor.py`
  (Further configuration options are available as global constants in the script).
"""

import os
import subprocess
import tempfile
import numpy as np
import librosa
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from typing import List, Dict, Optional # Added Optional
from app.services.services.download_youtube_video import download_video
import validators # For URL validation - this is a new dependency to note

# Attempt to import AI models; handle ImportError if they are not installed
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None # type: ignore 
    print("⚠️ WARNING: faster_whisper not installed. Transcription will be simulated.")

# Global variables for models (to be loaded once)
_stt_model = None
# _llm_model = None # Placeholder for future LLM model
# _vision_model = None # Placeholder for future Vision model

# ───────── CONFIGURACIÓN BÁSICA ───────── #
TARGET_CLIPS = 10
WIN_SIZE     = 1.0
MIN_CLIP_LEN = 60
MAX_CLIP_LEN = 90
AUDIO_W      = 2.0
MOTION_W     = 1.0

# ───────── AI MODEL CONFIGURATION ───────── #
MODEL_PATH_STT = "models/whisper-small" # Default path if downloading to local 'models' folder
# MODEL_PATH_LLM = "path/to/language_model"
# MODEL_PATH_VISION = "path/to/vision_model"
# ────────────────────────────────────────── #

# ───────── VIRALITY SCORING CONFIGURATION ───────── #
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

# ─── VRAM & EFFICIENCY STRATEGY (General Note) ─── #
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
# ──────────────────────────────────────── #

# ───────── MODEL LOADING FUNCTIONS ───────── #
def load_stt_model():
    global _stt_model
    if _stt_model is None and WhisperModel is not None:
        try:
            print(f"🧠 Loading STT model: {MODEL_PATH_STT}...")
            # This is the function to modify:
            _stt_model = WhisperModel(
                model_size_or_path="openai/whisper-small", # Changed as per instruction 1
                device="cuda",           # usa GPU
                compute_type="float16",       # o "float16" para más calidad
                local_files_only=False  # Added as per instruction 2
            )
            print("✅ STT Model loaded successfully.")
        except Exception as e:
            print(f"❌ ERROR loading STT model: {e}")
            print("Transcription will be simulated.")
            _stt_model = None # Ensure it's None if loading failed
    elif WhisperModel is None:
        print("ℹ️ faster_whisper library not available. STT model cannot be loaded. Transcription will be simulated.")

# def load_llm_model(): ...
# def load_vision_model(): ...
# ─────────────────────────────────────────── #


def audio_peaks(video_path: str, win: float) -> np.ndarray:
    # ... (function as before)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "44100", wav],
        capture_output=True, check=True,
    )
    y, sr = librosa.load(wav, sr=None)
    hop = int(sr * win)
    rms = librosa.feature.rms(y=y, frame_length=hop, hop_length=hop, center=False)[0]
    os.remove(wav)
    return rms

def motion_peaks(video_path: str, win: float) -> np.ndarray:
    # ... (function as before)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps * win)
    prev = None
    mvals = []
    idx = 0
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
    """
    Transcribe video audio to text with timestamps using the loaded Whisper model.
    Falls back to placeholder if model is not available.
    """
    if _stt_model is None:
        print(f"ℹ️  STT Model not loaded. Simulating transcription for {video_path}")
        return [
            {'start_time': 10.0, 'end_time': 15.0, 'text': 'This is a sample transcript segment.'},
            {'start_time': 16.0, 'end_time': 20.0, 'text': 'Another example sentence from the video.'},
        ]

    print(f"🎙️  Actual STT: Transcribing {video_path}...")
    audio_output_path = "" # Initialize to prevent NameError in finally if NamedTemporaryFile fails
    try:
        # Note: faster-whisper needs audio path, not video path directly.
        # Extract audio first
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            audio_output_path = tmp_audio.name
        
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", audio_output_path],
            capture_output=True, check=True
        )

        segments_generator, info = _stt_model.transcribe(audio_output_path, beam_size=5)
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        transcript_data = []
        for segment in segments_generator: # Changed variable name here
            transcript_data.append({
                "start_time": round(segment.start, 2),
                "end_time": round(segment.end, 2),
                "text": segment.text.strip()
            })
        
        print("✅ Actual STT: Transcription complete.")
        return transcript_data

    except Exception as e:
        print(f"❌ ERROR during actual STT transcription: {e}")
        print("Falling back to simulated transcription.")
        return [
            {'start_time': 10.0, 'end_time': 15.0, 'text': 'Error during transcription, using dummy data.'},
        ]
    finally:
        if audio_output_path and os.path.exists(audio_output_path): # Ensure cleanup even on error
            try: os.remove(audio_output_path)
            except Exception as e_rem: print(f"Error removing temp audio file {audio_output_path}: {e_rem}")


def analyze_text(transcript_segment: str, timestamp_info: Dict) -> Dict:
    # ... (function as before)
    analysis_results = {
        'timestamp_info': timestamp_info,
        'sentiment': {'label': 'neutral', 'score': 0.6}, 
        'emotions': ['curiosity', 'slight_interest'], 
        'keywords': ['sample', 'transcript', 'topic'],
        'topics': ['example_topic', 'discussion_point'],
        'humor_detected': False, 'controversy_detected': False,
        'engagement_hooks': ['question_found_example: what do you think?'],
        'summary': f"This segment from {timestamp_info['start']}s to {timestamp_info['end']}s appears to be about related keywords."
    }
    return analysis_results

def analyze_visuals(video_segment_path: str, timestamp_info: Dict) -> Dict:
    # ... (function as before)
    analysis_results = {
        'timestamp_info': timestamp_info,
        'facial_expressions': [{'person_id': 1, 'expression': 'neutral', 'confidence': 0.7, 'timestamp': timestamp_info['start'] + 1.0}],
        'detected_actions': [], 'key_objects': [],
        'scene_changes': [{'type': 'cut', 'timestamp': timestamp_info['start'] + 2.5}],
        'visual_intensity_score': 0.3,
        'notes': f"Visual analysis for segment at {video_segment_path} from {timestamp_info['start']}s to {timestamp_info['end']}s."
    }
    return analysis_results

def calculate_virality_score(
    text_analysis: Dict, visual_analysis: Dict,
    segment_audio_avg_rms: float, segment_motion_avg_score: float,
    config: Dict = VIRALITY_CONFIG
) -> float:
    # ... (function as before)
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
        for expr in visual_analysis.get('facial_expressions', []):
            if expr.get('expression') == 'happy' and expr.get('confidence', 0) > 0.6: score += weights.get('facial_expression_happy', 1.0)
        if len(visual_analysis.get('scene_changes', [])) > 2: score += weights.get('fast_cuts_or_action', 1.0)
    score += segment_audio_avg_rms * weights.get('audio_energy_avg', 0.1)
    score += segment_motion_avg_score * weights.get('motion_avg', 0.1)
    return round(score, 2)

def combined_score( # This function seems to be unused now, consider removing or refactoring if needed
    audio_energy: np.ndarray, motion_score: np.ndarray, aw: float, mw: float,
    text_analysis_results: Dict = None, visual_analysis_results: Dict = None
) -> np.ndarray:
    # ... (function as before)
    a = (audio_energy  - audio_energy.mean())  / (audio_energy.std()  + 1e-6)
    m = (motion_score - motion_score.mean()) / (motion_score.std() + 1e-6)
    n = min(len(a), len(m))
    current_score = aw * a[:n] + mw * m[:n]
    return current_score

def get_transcript_for_window(full_transcript: List[Dict], window_start_time: float, window_end_time: float) -> str:
    # ... (function as before)
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
    # ... (function as before, no changes to its internal logic here)
    if not audio_energy_per_second.any() or not motion_score_per_second.any(): print("⚠️ Warning: Audio or motion data is empty. Cannot select segments."); return []
    video_duration_seconds = len(audio_energy_per_second) * win_size
    if video_duration_seconds == 0: print("⚠️ Warning: Video duration is zero. Cannot select segments."); return []
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
        dummy_visual_path = f"dummy_path_for_segment_{current_start_time:.0f}_{current_end_time:.0f}.mp4"
        visual_analysis_results = analyze_visuals(dummy_visual_path, timestamp_info)
        start_idx = int(current_start_time / win_size); end_idx = int(current_end_time / win_size)
        valid_audio_slice = audio_energy_per_second[start_idx:end_idx] if start_idx < end_idx and start_idx < len(audio_energy_per_second) else np.array([0])
        valid_motion_slice = motion_score_per_second[start_idx:end_idx] if start_idx < end_idx and start_idx < len(motion_score_per_second) else np.array([0])
        avg_audio = np.mean(valid_audio_slice) if valid_audio_slice.any() else 0; avg_motion = np.mean(valid_motion_slice) if valid_motion_slice.any() else 0
        virality_score = calculate_virality_score(text_analysis_results, visual_analysis_results, avg_audio, avg_motion)
        potential_clips.append({'start': current_start_time, 'end': current_end_time, 'score': virality_score, 'duration': current_end_time - current_start_time})
    if not potential_clips: print("⚠️ No potential clips generated after analysis."); return []
    potential_clips.sort(key=lambda x: x['score'], reverse=True); selected_clips = []; chosen_intervals = []
    for clip_candidate in potential_clips:
        if len(selected_clips) >= nclips: break
        cs = clip_candidate['start']; ce = clip_candidate['end']; cd = clip_candidate['duration']
        # if cd < min_len or cd > max_len: pass # This logic seems to be bypassed
        is_overlapping = False
        for chosen_s, chosen_e in chosen_intervals:
            if cs < chosen_e and chosen_s < ce: is_overlapping = True; break
        if not is_overlapping: selected_clips.append({'start': cs, 'end': ce, 'score': clip_candidate['score']}); chosen_intervals.append((cs, ce))
    print(f"🏆 Selected {len(selected_clips)} clips out of {len(potential_clips)} potential clips.")
    return selected_clips

def cut_clips(video_path: str, segments: List[Dict[str, float]], outdir="clips"):
    # ... (function as before)
    os.makedirs(outdir, exist_ok=True); outputs = []
    for k, seg_info in enumerate(segments, 1):
        s = seg_info['start']; e = seg_info['end']; name = os.path.join(outdir, f"clip{k}.mp4")
        cmd = ["ffmpeg", "-y", "-i", video_path, "-ss", str(s), "-to", str(e), "-vf", "format=yuv420p",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "128k",
               "-movflags", "+faststart", name]
        subprocess.run(cmd, check=True, capture_output=True)
        outputs.append(name)
    return outputs

def main():
    """
    Main function to orchestrate the video processing pipeline.
    Handles input (local file or YouTube URL), downloads if necessary,
    then runs transcription, audio/visual analysis, segment selection,
    and clip cutting.
    """
    load_stt_model() # Attempt to load STT model at startup
    # load_llm_model() # Placeholder
    # load_vision_model() # Placeholder

    video_input_source = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Example YouTube URL
    video_path_for_processing = None
    print(f"Processing input: {video_input_source}")
    if validators.url(video_input_source):
        print(f"Input identified as URL. Attempting download...")
        download_output_dir = "app/media" 
        downloaded_path = download_video(video_input_source, output_dir=download_output_dir)
        if downloaded_path: video_path_for_processing = downloaded_path; print(f"Video ready for processing: {video_path_for_processing}")
        else: print(f"Failed to download video from URL: {video_input_source}. Exiting."); return
    else:
        print(f"Input identified as local file path: {video_input_source}")
        if os.path.exists(video_input_source): video_path_for_processing = video_input_source; print(f"Local video found: {video_path_for_processing}")
        else: print(f"Local video file not found: {video_input_source}. Exiting."); return
    if not video_path_for_processing: print("No valid video to process. Exiting."); return

    try:
        print("🎙️ Transcribing video...")
        full_transcript = transcribe_video(video_path_for_processing) # Uses loaded _stt_model or simulates
        print("✅ Transcription complete.")
    except Exception as e: print(f"❌ ERROR during transcription: {e}\nCould not perform transcription. Exiting."); return

    try: print("⏳ Analyzing audio..."); audio_rms = audio_peaks(video_path_for_processing, WIN_SIZE); print("✅ Audio analysis complete.")
    except Exception as e: print(f"❌ ERROR during audio analysis: {e}\nCould not perform audio analysis. Exiting."); return

    try: print("⏳ Analyzing video for motion..."); motion = motion_peaks(video_path_for_processing, WIN_SIZE); print("✅ Motion analysis complete.")
    except Exception as e: print(f"❌ ERROR during motion analysis: {e}\nCould not perform motion analysis. Exiting."); return

    segments = []
    try:
        print("🧠 Identifying best segments with AI (placeholders & loaded STT)...")
        segments = best_segments(video_path_for_processing, full_transcript, audio_rms, motion, WIN_SIZE, TARGET_CLIPS, MIN_CLIP_LEN, MAX_CLIP_LEN)
        print("✅ Segment identification complete.")
    except Exception as e: print(f"❌ ERROR during segment identification: {e}\nCould not identify best segments. Exiting."); return
    
    files = []
    print("🔖 Segmentos seleccionados (Top " + str(TARGET_CLIPS) + "):")
    if segments:
        for i, seg_info in enumerate(segments):
            start_time_str = f"{seg_info['start']:.2f}"; end_time_str = f"{seg_info['end']:.2f}"; virality_score_str = f"{seg_info['score']:.2f}"
            print(f"Clip #{i+1}: [{start_time_str}s - {end_time_str}s], Virality Score: {virality_score_str}")
        try:
            print("✂️  Cutting clips..."); files = cut_clips(video_path_for_processing, segments) 
            print("✅ Clips generated successfully.")
            if not files: print("⚠️ WARNING: `cut_clips` executed but returned no file paths. Clips might be missing.")
            else: print(f"🎞️ Generated clip files: {files}")
        except Exception as e: print(f"❌ ERROR during clip cutting: {e}\nCould not cut clips.")
    else: print("No segments were selected by `best_segments`.")
    if not files: print("⚠️ FINAL WARNING: No clips were ultimately generated or saved from the video.")

if __name__ == "__main__":
    main()

# mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm #
# --------------------- CONCEPTUAL TESTING STRATEGY ----------------------- #
# ... (comment block as before)
# mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm #
```
