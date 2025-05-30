# NOTES FOR README.md

## Project Overview
`clipExtractor.py` is a Python script designed to automatically identify and extract potentially viral short clips (target 30-90 seconds) from longer videos. It leverages multi-modal analysis, combining speech transcription, Large Language Model (LLM) based content understanding, visual analysis (object and emotion detection), and audio-visual energy/motion cues.

## Core Features & Workflow
1.  **Video Input**: Handles local video files and YouTube URLs (downloads automatically).
2.  **Transcription**: Uses `faster-whisper` for accurate speech-to-text with word-level timestamps.
3.  **LLM Candidate Identification**:
    *   Employs an advanced LLM strategy (e.g., using `google/gemma-2b-it` as a fallback) to analyze the entire video transcript.
    *   For very long transcripts that might exceed the LLM's context window, the transcript is automatically split into overlapping chunks.
    *   LLM calls for these chunks are parallelized using `asyncio` for improved performance.
    *   The LLM is prompted to act as an expert video editor and suggest highlight-worthy segments with start/end times, themes, and reasons.
4.  **Multi-Modal Segment Scoring**:
    *   **Text Analysis**: Themes and reasons from LLM suggestions are processed.
    *   **Visual Analysis**: For each LLM-suggested segment, frames are sampled. YOLOv5 detects key objects, and a Facial Emotion Recognition (FER) model identifies dominant emotions.
    *   **Audio/Motion Analysis**: Average audio energy (RMS) and motion scores (from optical flow between frames, optimized with batch GPU processing) are calculated for each segment.
    *   **Virality Score**: A weighted score is computed based on all these factors (text, visuals, audio, motion) to determine the 'virality potential' of each LLM-suggested segment.
5.  **Clip Selection & Extraction**:
    *   Segments are ranked by their virality score.
    *   The top `N` (default 10) non-overlapping segments are selected, prioritizing higher scores. Segment times are adjusted if possible to meet duration constraints (30-90s).
    *   `ffmpeg` is used to cut these selected clips from the original video.
6.  **Performance Logging**:
    *   The script logs detailed performance metrics for major operations: transcription time, LLM analysis time (total and per chunk if applicable), motion analysis FPS, VRAM usage (if GPU is used), clip cutting time, etc. This helps in assessing and optimizing performance.

## Usage
1.  **Setup**:
    *   Ensure Python 3.8+ is installed.
    *   Install all dependencies: `pip install -r requirements.txt`.
    *   Ensure `ffmpeg` is installed and accessible in your system's PATH.
2.  **Running the Script**:
    *   The script is executed asynchronously.
    *   Command: `python clipExtractor.py <video_input_source> [options]`
    *   `<video_input_source>`: Can be a local file path (e.g., `"my_video.mp4"`) or a YouTube URL (e.g., `"https://www.youtube.com/watch?v=dQw4w9WgXcQ"`).
    *   Options:
        *   `--debug`: Enable debug level logging.
        *   `-o OUTPUT_DIR`, `--output-dir OUTPUT_DIR`: Specify the directory to save extracted clips (default: `"clips/"`).
    *   Example: `python clipExtractor.py "https://www.youtube.com/watch?v=your_video_id" -o my_awesome_clips`

## Configuration
(Details about key configuration constants like `MIN_CLIP_LEN`, `MAX_CLIP_LEN`, `TARGET_CLIPS`, model paths, `VIRALITY_CONFIG` weights can be added here if they are intended to be user-configurable or for advanced users to tweak.)

## Models
*   **Speech-to-Text (STT)**: `faster-whisper` (e.g., "small" model). Models are downloaded automatically on first use.
*   **Large Language Model (LLM)**:
    *   Primary: `meta-llama/Llama-2-7b-chat-hf` (requires Hugging Face token and GPU).
    *   Fallback (Open Source): `google/gemma-2b-it`.
    *   Fallback (Small): `distilgpt2`.
    *   Models are downloaded from Hugging Face Hub and cached locally in the `models/` directory.
*   **Vision**:
    *   Object Detection: YOLOv5 (`yolov5s.pt`). Downloaded if not present.
    *   Facial Emotion Recognition: `fer` library (MTCNN for face detection).
*   Model paths and quantization configurations (like BitsAndBytes for 4-bit LLM loading) are defined within the script.

## Note on Large File Storage (git-lfs)
*   The script downloads necessary AI models (STT, LLM, YOLO) from their sources (like Hugging Face Hub) into a local `models/` directory at runtime if they are not already present.
*   Therefore, `git-lfs` is generally **not** required for these operational models unless you intend to commit the downloaded model files (which can be very large) directly into the Git repository. It's usually better to let the script manage model downloads.
*   If you add other large asset files directly to the repository, then `git-lfs` would be appropriate for those.
