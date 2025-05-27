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
  numpy, librosa, opencv-python, moviepy, yt-dlp, validators, ffmpeg (system path).

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
from typing import List, Dict
from app.services.services.download_youtube_video import download_video
import validators # For URL validation - this is a new dependency to note
# Ensure 'os' is imported if not already (it should be for cut_clips)

# ───────── CONFIGURACIÓN BÁSICA ───────── #
# VIDEO_PATH   = "input.mp4"   # ruta al vídeo de prueba (~3 min) # Now handled in main
TARGET_CLIPS = 10            # clips a extraer
WIN_SIZE     = 1.0           # ventana de análisis (s)
MIN_CLIP_LEN = 60  # minimum duration of each clip (s)
MAX_CLIP_LEN = 90  # maximum duration of each clip (s)
AUDIO_W      = 2.0           # peso del audio en la puntuación
MOTION_W     = 1.0           # peso del movimiento
# ───────── AI MODEL CONFIGURATION (Placeholders) ───────── #
# MODEL_PATH_STT = "path/to/stt_model"                # Path to Speech-to-Text model
# MODEL_PATH_LLM = "path/to/language_model"           # Path to Large Language Model
# MODEL_PATH_VISION = "path/to/vision_model"          # Path to Vision Model
#
# # Example thresholds or model-specific settings
# CONF_THRESHOLD_SENTIMENT = 0.7 # Minimum confidence for sentiment detection
# LLM_MAX_TOKENS = 512           # Max tokens for LLM processing per segment
# ────────────────────────────────────────────────────────── #

# ───────── VIRALITY SCORING CONFIGURATION ───────── #
VIRALITY_CONFIG = {
    'weights': {
        'sentiment_positive': 1.5, # Weight for positive sentiment
        'sentiment_negative': 0.8, # Negative sentiment might be viral for controversy
        'emotion_joy': 1.2,
        'emotion_surprise': 1.3,
        'keyword_match': 2.0,    # Weight if relevant keywords are found
        'engagement_hook': 1.5,  # Weight for questions, CTAs
        'visual_intensity': 1.0,
        'facial_expression_happy': 1.2,
        'fast_cuts_or_action': 1.1, # If scene_changes indicate action
        'audio_energy_avg': 0.5, # Average audio energy of the segment
        'motion_avg': 0.3,       # Average motion in the segment
    },
    'thresholds': {
        'min_sentiment_score': 0.5,
        'min_visual_intensity': 0.2,
    },
    'trending_keywords': ['challenge', 'hack', 'reveal', 'shocking', 'must-see'] # Example
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

def audio_peaks(video_path: str, win: float) -> np.ndarray:
    """Devuelve energía RMS por ventana.
    # For long videos, this could be used to identify overall 'active audio zones'
    # to narrow down segments for more detailed AI processing.
    """
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
    """Devuelve cantidad de movimiento medio por ventana.
    # For long videos, this could identify 'high motion zones'.
    # These zones could be prioritized or combined with audio activity
    # for selecting segments for deeper AI analysis.
    """
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


def analyze_text(transcript_segment: str, timestamp_info: Dict) -> Dict:
    """Analyzes a text segment for sentiment, keywords, topics, humor, controversy, and engagement hooks. Placeholder for actual LLM/NLP model integration."""
    # VRAM_NOTE: Ensure LLM is loaded/used efficiently. Consider techniques like:
    # - Model quantization (e.g., GGUF for llama.cpp, ONNX quantization).
    # - Offloading parts of the model to CPU if VRAM is insufficient.
    # - Using smaller, fine-tuned models specific to the tasks (sentiment, keywords).
    # TODO: Implement actual text analysis using an LLM or NLP libraries.
    # - Sentiment Analysis (e.g., positive, negative, neutral, joy, anger)
    # - Keyword/Topic Extraction
    # - Humor/Controversy Detection (e.g., using keyword spotting or specialized models)
    # - Engagement Hook Identification (e.g., questions, exclamations, calls to action)
    
    # Dummy results (replace with actual model outputs)
    analysis_results = {
        'timestamp_info': timestamp_info,
        'sentiment': {'label': 'neutral', 'score': 0.6}, # Example: 'positive', 'negative'
        'emotions': ['curiosity', 'slight_interest'], # Example granular emotions
        'keywords': ['sample', 'transcript', 'topic'],
        'topics': ['example_topic', 'discussion_point'],
        'humor_detected': False,
        'controversy_detected': False,
        'engagement_hooks': ['question_found_example: what do you think?'],
        'summary': f"This segment from {timestamp_info['start']}s to {timestamp_info['end']}s appears to be about related keywords."
    }
    return analysis_results


def analyze_visuals(video_segment_path: str, timestamp_info: Dict) -> Dict:
    """Analyzes a video segment for key visual moments like expressions, actions, and scene changes. Placeholder for actual vision model integration."""
    # VRAM_NOTE: Vision models can be VRAM intensive. Consider:
    # - Unloading other large models (like LLMs) from VRAM if running sequentially.
    # - Using smaller architectures or quantized versions (e.g., ONNX).
    # - Processing frames at a lower resolution or frame rate if acceptable.
    # TODO: Implement actual visual analysis using vision models.
    # - Facial Expression Recognition (e.g., happy, sad, surprised)
    # - Action Recognition / Object Detection (e.g., specific actions, important objects)
    # - Scene Change Detection / Fast Cuts / Dramatic Pauses
    
    # Dummy results (replace with actual model outputs)
    analysis_results = {
        'timestamp_info': timestamp_info,
        'facial_expressions': [{'person_id': 1, 'expression': 'neutral', 'confidence': 0.7, 'timestamp': timestamp_info['start'] + 1.0}],
        'detected_actions': [], # e.g., ['hand_wave', 'pointing']
        'key_objects': [],     # e.g., ['book', 'computer']
        'scene_changes': [{'type': 'cut', 'timestamp': timestamp_info['start'] + 2.5}], # Could also be 'fade', 'dissolve'
        'visual_intensity_score': 0.3, # Placeholder for overall visual activity or interest
        'notes': f"Visual analysis for segment at {video_segment_path} from {timestamp_info['start']}s to {timestamp_info['end']}s."
    }
    # In a real scenario, you might need to extract frames from video_segment_path
    # or use a library that can process video files directly.
    return analysis_results


def calculate_virality_score(
    text_analysis: Dict,
    visual_analysis: Dict,
    segment_audio_avg_rms: float,
    segment_motion_avg_score: float,
    config: Dict = VIRALITY_CONFIG  # Use the global config by default
) -> float:
    """Calculates a virality score for a segment based on text, visual, audio, and motion analysis. This is a placeholder and will need significant tuning."""
    score = 0.0
    weights = config.get('weights', {})
    thresholds = config.get('thresholds', {})
    trending_keywords = config.get('trending_keywords', [])

    # Text analysis contributions
    if text_analysis:
        # Sentiment scoring (example)
        sentiment_label = text_analysis.get('sentiment', {}).get('label', 'neutral')
        sentiment_score_val = text_analysis.get('sentiment', {}).get('score', 0.0) # Renamed to avoid conflict
        if sentiment_score_val > thresholds.get('min_sentiment_score', 0.3):
            if sentiment_label == 'positive':
                score += sentiment_score_val * weights.get('sentiment_positive', 1.0)
            elif sentiment_label == 'negative': # Negative can also be viral (controversy)
                score += sentiment_score_val * weights.get('sentiment_negative', 0.5)
        
        # Emotion scoring (example)
        for emotion in text_analysis.get('emotions', []):
            score += weights.get(f'emotion_{emotion}', 0.0) # e.g., emotion_joy

        # Keyword scoring (example)
        found_keywords = text_analysis.get('keywords', [])
        for kw in found_keywords:
            if kw in trending_keywords:
                score += weights.get('keyword_match', 1.0)
        
        # Engagement hook (example)
        if text_analysis.get('engagement_hooks'):
            score += weights.get('engagement_hook', 1.0)

    # Visual analysis contributions
    if visual_analysis:
        visual_intensity = visual_analysis.get('visual_intensity_score', 0.0)
        if visual_intensity > thresholds.get('min_visual_intensity', 0.1):
            score += visual_intensity * weights.get('visual_intensity', 1.0)

        # Facial expression (example - very simplified)
        for expr in visual_analysis.get('facial_expressions', []):
            if expr.get('expression') == 'happy' and expr.get('confidence', 0) > 0.6:
                score += weights.get('facial_expression_happy', 1.0)
        
        # Scene changes / action (example)
        if len(visual_analysis.get('scene_changes', [])) > 2: # More than 2 cuts could mean action
             score += weights.get('fast_cuts_or_action', 1.0)


    # Audio and Motion contributions (using average values for the segment)
    score += segment_audio_avg_rms * weights.get('audio_energy_avg', 0.1)
    score += segment_motion_avg_score * weights.get('motion_avg', 0.1)
    
    # TODO: Add more sophisticated interaction terms. 
    # e.g., high sentiment + high visual intensity might be weighted more than sum of parts.
    # TODO: Normalize scores from different sources if their ranges are very different.

    return round(score, 2) # Return a rounded score


def combined_score(
    audio_energy: np.ndarray,
    motion_score: np.ndarray,
    aw: float,
    mw: float,
    text_analysis_results: Dict = None, # Placeholder for text AI output
    visual_analysis_results: Dict = None # Placeholder for vision AI output
) -> np.ndarray:
    """
    Combines various scores including audio energy, motion, and future AI analyses
    to produce a segment importance score.
    """
    # TODO: Incorporate text_analysis_results (sentiment, keywords, etc.)
    # For example: score += text_analysis_results.get('sentiment_score', 0) * text_weight

    # TODO: Incorporate visual_analysis_results (expressions, scene changes, etc.)
    # For example: score += visual_analysis_results.get('visual_intensity', 0) * vision_weight

    # Current base score (will be part of a more complex calculation)
    a = (audio_energy  - audio_energy.mean())  / (audio_energy.std()  + 1e-6)
    m = (motion_score - motion_score.mean()) / (motion_score.std() + 1e-6)
    n = min(len(a), len(m))
    current_score = aw * a[:n] + mw * m[:n]
    return current_score # This will eventually be a more comprehensive score


def get_transcript_for_window(full_transcript: List[Dict], window_start_time: float, window_end_time: float) -> str:
    """Extracts and concatenates transcript text within a given time window."""
    segment_texts = []
    for entry in full_transcript:
        # Check for overlap between transcript entry and window
        if entry['start_time'] < window_end_time and entry['end_time'] > window_start_time:
            segment_texts.append(entry['text'])
    return " ".join(segment_texts)


def best_segments(
    video_path: str,
    full_transcript: List[Dict],
    audio_energy_per_second: np.ndarray,
    motion_score_per_second: np.ndarray,
    win_size: float, # Interval of audio/motion scores
    nclips: int = TARGET_CLIPS,
    min_len: int = MIN_CLIP_LEN,
    max_len: int = MAX_CLIP_LEN
) -> List[Dict]:
    """
    Identifies the best N clips based on windowed AI analysis and virality scoring.
    Returns a list of dicts, each with 'start', 'end', 'score'.
    """
    if not audio_energy_per_second.any() or not motion_score_per_second.any():
        print("⚠️ Warning: Audio or motion data is empty. Cannot select segments.")
        return []

    video_duration_seconds = len(audio_energy_per_second) * win_size
    if video_duration_seconds == 0:
        print("⚠️ Warning: Video duration is zero. Cannot select segments.")
        return []

    analysis_window_duration = (min_len + max_len) // 2
    if analysis_window_duration <= 0:
        analysis_window_duration = 60 # Default if min/max_len are too small or invalid
        print(f"⚠️ Warning: Calculated analysis_window_duration was <=0. Defaulted to {analysis_window_duration}s.")

    step_size = max(15, analysis_window_duration // 4) # Ensure step_size is at least 15s

    potential_clips = []
    print(f"⚙️  Analyzing video of {video_duration_seconds:.2f}s. Window: {analysis_window_duration}s, Step: {step_size}s")

    for current_start_time in np.arange(0, video_duration_seconds - analysis_window_duration + 1, step_size):
        current_end_time = current_start_time + analysis_window_duration
        if current_end_time > video_duration_seconds: # Ensure window doesn't exceed video
            current_end_time = video_duration_seconds
            if current_start_time >= current_end_time : continue # skip if window is zero or negative

        # a. Extract Transcript Segment
        transcript_for_window = get_transcript_for_window(full_transcript, current_start_time, current_end_time)
        timestamp_info = {'start': current_start_time, 'end': current_end_time}

        # b. Call analyze_text
        text_analysis_results = analyze_text(transcript_for_window, timestamp_info)

        # EFFICIENCY_NOTE for analyze_visuals:
        # In a real implementation, repeatedly creating/passing `video_segment_path`
        # for each window implies extracting many subclips. This can be I/O intensive.
        # More efficient alternatives for visual analysis could involve:
        # - Processing the entire video stream once with the vision model to detect events.
        # - Extracting frames for relevant segments and passing them as batches.
        # The current dummy path `f"dummy_path_for_segment_{...}.mp4"` highlights this.
        # c. Call analyze_visuals
        dummy_visual_path = f"dummy_path_for_segment_{current_start_time:.0f}_{current_end_time:.0f}.mp4"
        visual_analysis_results = analyze_visuals(dummy_visual_path, timestamp_info)

        # d. Calculate Average Audio/Motion for Window
        start_idx = int(current_start_time / win_size)
        end_idx = int(current_end_time / win_size)
        
        # Ensure indices are within bounds and start_idx < end_idx for slicing
        valid_audio_slice = audio_energy_per_second[start_idx:end_idx] if start_idx < end_idx and start_idx < len(audio_energy_per_second) else np.array([0])
        valid_motion_slice = motion_score_per_second[start_idx:end_idx] if start_idx < end_idx and start_idx < len(motion_score_per_second) else np.array([0])

        avg_audio = np.mean(valid_audio_slice) if valid_audio_slice.any() else 0
        avg_motion = np.mean(valid_motion_slice) if valid_motion_slice.any() else 0
        
        # e. Call calculate_virality_score
        virality_score = calculate_virality_score(text_analysis_results, visual_analysis_results, avg_audio, avg_motion)
        
        # f. Store Results
        potential_clips.append({
            'start': current_start_time,
            'end': current_end_time,
            'score': virality_score,
            'duration': current_end_time - current_start_time # Actual duration of this window
        })
        # print(f"  Window {current_start_time:.1f}s-{current_end_time:.1f}s, Transcript: '{transcript_for_window[:50]}...', Score: {virality_score:.2f}")


    # Rank and Select Top N Non-Overlapping Clips
    if not potential_clips:
        print("⚠️ No potential clips generated after analysis.")
        return []
        
    potential_clips.sort(key=lambda x: x['score'], reverse=True)

    selected_clips = []
    # For non-overlapping, we can mark time slots. Video duration in seconds.
    # Or, more simply, a list of (start, end) tuples for chosen clips.
    chosen_intervals = [] 

    for clip_candidate in potential_clips:
        if len(selected_clips) >= nclips:
            break

        cs = clip_candidate['start']
        ce = clip_candidate['end']
        cd = clip_candidate['duration']

        # Validate against min_len and max_len more strictly here if needed,
        # though analysis_window_duration is already based on them.
        # For this iteration, we assume analysis_window_duration is the target clip length.
        if cd < min_len or cd > max_len: # If strict adherence to min/max is needed for final clips
            # This check is somewhat redundant if analysis_window_duration is used as clip duration
            # and it's derived from min_len/max_len. But useful if window duration could vary.
            # print(f"  Skipping candidate {cs}-{ce} (score {clip_candidate['score']}) due to duration {cd}s not in [{min_len}, {max_len}]")
            pass # For now, we accept the analysis_window_duration as the clip duration

        is_overlapping = False
        for chosen_s, chosen_e in chosen_intervals:
            # Check for overlap: (s1 < e2) and (s2 < e1)
            if cs < chosen_e and chosen_s < ce:
                is_overlapping = True
                break
        
        if not is_overlapping:
            selected_clips.append({
                'start': cs,
                'end': ce,
                'score': clip_candidate['score']
            })
            chosen_intervals.append((cs, ce))
            # print(f"  Selected clip {cs:.1f}s-{ce:.1f}s, Score: {clip_candidate['score']:.2f}")

    print(f"🏆 Selected {len(selected_clips)} clips out of {len(potential_clips)} potential clips.")
    return selected_clips


def cut_clips(video_path: str, segments: List[Dict[str, float]], outdir="clips"):
    os.makedirs(outdir, exist_ok=True)
    outputs = []
    for k, seg_info in enumerate(segments, 1):
        s = seg_info['start']
        e = seg_info['end']
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


def transcribe_video(video_path: str) -> List[Dict]:
    """
    Transcribe video audio to text with timestamps.
    This is a placeholder function.
    """
    # TODO: Implement actual STT logic using a library like Whisper, Vosk, or an API.
    print(f"ℹ️  Placeholder STT: Simulating transcription for {video_path}")
    return [
        {'start_time': 10.0, 'end_time': 15.0, 'text': 'This is a sample transcript segment.'},
        {'start_time': 16.0, 'end_time': 20.0, 'text': 'Another example sentence from the video.'},
        {'start_time': 22.0, 'end_time': 28.0, 'text': 'This part discusses an interesting topic.'},
        {'start_time': 30.0, 'end_time': 35.0, 'text': 'And here is a call to action or a question.'},
        {'start_time': 36.0, 'end_time': 40.0, 'text': 'More speech detected here.'},
        {'start_time': 42.0, 'end_time': 48.0, 'text': 'Final segment of this dummy transcript.'}
    ]

def main():
    # ───────── AI MODEL LOADING (Placeholders) ───────── #
    # TODO: Implement actual model loading functions.
    # These would ideally be loaded once if memory allows.
    # Example:
    # stt_model = load_stt_model(MODEL_PATH_STT) # Replace with actual STT loading
    # llm_model = load_llm_model(MODEL_PATH_LLM)   # Replace with actual LLM loading
    # vision_model = load_vision_model(MODEL_PATH_VISION) # Replace with actual Vision model loading
    # print("🧠 AI Models (placeholders) would be loaded here if they were implemented.")
    # ─────────────────────────────────────────────────── #

    """
    Main function to orchestrate the video processing pipeline.
    Handles input (local file or YouTube URL), downloads if necessary,
    then runs transcription, audio/visual analysis, segment selection,
    and clip cutting.
    """
    # video_input_source = "input.mp4"  # Example for local file
    video_input_source = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Example YouTube URL
    # This could also be made an argument to main() or read from config/argv in a fuller app
    
    video_path_for_processing = None

    print(f"Processing input: {video_input_source}")

    if validators.url(video_input_source):
        print(f"Input identified as URL. Attempting download...")
        download_output_dir = "app/media" 
        
        downloaded_path = download_video(video_input_source, output_dir=download_output_dir)
        if downloaded_path:
            video_path_for_processing = downloaded_path
            print(f"Video ready for processing: {video_path_for_processing}")
        else:
            print(f"Failed to download video from URL: {video_input_source}. Exiting.")
            return # Exit main if download fails
    else:
        print(f"Input identified as local file path: {video_input_source}")
        if os.path.exists(video_input_source):
            video_path_for_processing = video_input_source
            print(f"Local video found: {video_path_for_processing}")
        else:
            print(f"Local video file not found: {video_input_source}. Exiting.")
            return # Exit main if local file not found
    
    if not video_path_for_processing:
        print("No valid video to process. Exiting.")
        return

    try:
        print("🎙️ Transcribing video (placeholder)...")
        full_transcript = transcribe_video(video_path_for_processing)
        print("✅ Transcription (placeholder) complete.")
    except Exception as e:
        print(f"❌ ERROR during transcription (placeholder): {e}")
        print("Could not perform transcription. Exiting.")
        return

    try:
        print("⏳ Analyzing audio...")
        audio_rms = audio_peaks(video_path_for_processing, WIN_SIZE)
        print("✅ Audio analysis complete.")
    except Exception as e:
        print(f"❌ ERROR during audio analysis: {e}")
        print("Could not perform audio analysis. Exiting.")
        return

    try:
        print("⏳ Analyzing video for motion...")
        motion = motion_peaks(video_path_for_processing, WIN_SIZE)
        print("✅ Motion analysis complete.")
    except Exception as e:
        print(f"❌ ERROR during motion analysis: {e}")
        print("Could not perform motion analysis. Exiting.")
        return

    # print("⏳ Combinando puntuaciones...") # This was for the old combined_score, can be removed
    # The old combined_score is no longer the primary source for best_segments' first parameter
    # score = combined_score(audio_rms, motion, AUDIO_W, MOTION_W, text_analysis_results=None, visual_analysis_results=None)

    segments = [] # Initialize segments
    try:
        print("🧠 Identifying best segments with AI (placeholders)...")
        segments = best_segments(
            video_path_for_processing,
            full_transcript,
            audio_rms,
            motion,
            WIN_SIZE,
            TARGET_CLIPS,
            MIN_CLIP_LEN,
            MAX_CLIP_LEN
        )
        print("✅ Segment identification complete.")
    except Exception as e:
        print(f"❌ ERROR during segment identification: {e}")
        print("Could not identify best segments. Exiting.")
        return
    
    files = [] # Initialize files
    print("🔖 Segmentos seleccionados (Top " + str(TARGET_CLIPS) + "):")
    if segments:
        for i, seg_info in enumerate(segments):
            start_time_str = f"{seg_info['start']:.2f}"
            end_time_str = f"{seg_info['end']:.2f}"
            virality_score_str = f"{seg_info['score']:.2f}"
            print(f"Clip #{i+1}: [{start_time_str}s - {end_time_str}s], Virality Score: {virality_score_str}")
        
        try:
            print("✂️  Cutting clips...")
            files = cut_clips(video_path_for_processing, segments) 
            print("✅ Clips generated successfully.")
            if not files:
                print("⚠️ WARNING: `cut_clips` executed but returned no file paths. Clips might be missing.")
            else:
                print(f"🎞️ Generated clip files: {files}")

        except Exception as e:
            print(f"❌ ERROR during clip cutting: {e}")
            # files is already initialized to [] or contains results from cut_clips
            print("Could not cut clips.") 
            # Not returning here, so it can still print the "No clips were generated" if applicable

    else:
        print("No segments were selected by `best_segments`.")
        # files is already initialized to []

    # Final check
    if not files: 
        print("⚠️ FINAL WARNING: No clips were ultimately generated or saved from the video.")


if __name__ == "__main__":
    main()

# mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm #
# --------------------- CONCEPTUAL TESTING STRATEGY ----------------------- #
# mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm #
#
# This script, in its current state, uses placeholders for core AI functionalities.
# Testing will focus on data flow, logic with dummy data, and output format.
#
# 1. PREPARATION:
#    - Prepare a short test video (e.g., 1-5 minutes). Ensure VIDEO_PATH points to it.
#    - Alternatively, for a quick "dry run" without actual video processing for
#      audio/motion, one could temporarily modify main() to use dummy np.arrays
#      for audio_rms and motion if VIDEO_PATH does not exist or if ffmpeg is not setup.
#
# 2. UNIT TESTS (Conceptual - would be separate test files in practice):
#    - `get_transcript_for_window(full_transcript, start, end)`:
#      - Test with various window overlaps, edge cases (start/end of video).
#      - Ensure it correctly concatenates text from the dummy `full_transcript`.
#    - `calculate_virality_score(text_analysis, visual_analysis, ...)`:
#      - Given known dummy `text_analysis` and `visual_analysis` dicts, and audio/motion
#        averages, check if the score is calculated as expected based on VIRALITY_CONFIG.
#      - Test edge cases (e.g., empty analysis dicts, zero scores/weights).
#
# 3. INTEGRATION TEST (Running the main script):
#    - **Data Flow Verification:**
#      - `transcribe_video`: Ensure its dummy transcript is generated and passed along.
#      - `audio_peaks`, `motion_peaks`: Ensure they run (if video exists) or are bypassed gracefully.
#        Their outputs should feed into `best_segments`.
#      - `analyze_text`, `analyze_visuals`: Verify (e.g., via temporary print statements
#        or a debugger) that they are called for analysis windows and their dummy
#        structured data is returned.
#      - `calculate_virality_score`: Check it's called for each window and uses data from
#        the AI placeholder functions.
#    - **`best_segments` Logic:**
#      - **Segment Count:** Does it attempt to find `TARGET_CLIPS` (e.g., 10)?
#        (Actual number may be less if video is short or dummy scores are not diverse).
#      - **Segment Duration:** Are selected segments based on `analysis_window_duration`
#        which uses `MIN_CLIP_LEN`, `MAX_CLIP_LEN`?
#      - **Non-Overlapping:** Visually inspect start/end times of outputted segments.
#      - **Score Usage:** Are segments generally ordered by the dummy virality scores?
#    - **`cut_clips` Execution (if test video is provided):**
#      - Does FFmpeg get called without errors for the selected segments?
#      - Are clip files created in the `clips/` directory?
#        (Content of clips will be based on dummy scores, so not semantically relevant yet).
#    - **Output Format:**
#      - Does the console output match the specified format:
#        `Clip #X: [start_time_str_s - end_time_str_s], Virality Score: virality_score_str`?
#
# 4. CONFIGURATION TESTING:
#    - Modify `TARGET_CLIPS` (e.g., to 3) and check if output reflects this.
#    - Modify `MIN_CLIP_LEN`, `MAX_CLIP_LEN` and observe changes in segment durations
#      (via `analysis_window_duration`).
#    - Temporarily set all weights in `VIRALITY_CONFIG` to 0. All virality scores
#      should become 0 (or close to it, depending on base audio/motion scores if they
#      are not weighted to zero).
#
# 5. MANUAL REVIEW:
#    - Read through the code, checking for logical flow and clarity of comments,
#      especially around placeholder sections.
#
# Actual AI model accuracy testing would occur *after* real models are integrated.
# That would involve a different dataset, metrics (e.g., precision/recall for
# specific content types), and qualitative human evaluation.
#
# mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm mMmmMm #
