import pytest
import os
import numpy as np
from scipy.io import wavfile
import tempfile

# Adjust import paths as necessary
from app.services.services.clipExtractor import load_stt_model, transcribe_video, DEVICE, _stt_model as clipExtractor_stt_model, MODEL_PATH_STT
from faster_whisper import WhisperModel # For mocking its constructor

# Mock the logger
@pytest.fixture
def mock_logger(mocker):
    return mocker.patch('app.services.services.clipExtractor.log')

@pytest.fixture(scope="module")
def dummy_wav_file():
    """Create a dummy WAV file for testing and return its path."""
    sample_rate = 16000
    duration = 1  # seconds
    frequency = 440  # Hz (A4 note)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Ensure a temporary file is created with .wav extension
    # and that it's properly closed before whisper tries to read it.
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavfile.write(temp_file.name, sample_rate, wave.astype(np.int16))
    temp_file.close() # Close the file so it can be opened by other processes
    
    yield temp_file.name # Provide the path to the test
    
    os.unlink(temp_file.name) # Clean up after tests in this module are done

@pytest.fixture(autouse=True)
def reset_stt_model_global():
    """Fixture to reset the global _stt_model in clipExtractor before/after each test."""
    global clipExtractor_stt_model
    original_model = clipExtractor_stt_model
    clipExtractor_stt_model = None # Reset before test
    yield
    clipExtractor_stt_model = original_model # Restore after test (though None is typical starting state)


def test_load_stt_model_success(mocker, mock_logger):
    """Test successful STT model loading (mocked)."""
    global clipExtractor_stt_model # We are testing the effect on the global variable
    
    mock_whisper_constructor = mocker.patch('faster_whisper.WhisperModel', autospec=True)
    mock_model_instance = mocker.MagicMock(spec=WhisperModel)
    mock_whisper_constructor.return_value = mock_model_instance

    load_stt_model() # This should set the global _stt_model

    mock_whisper_constructor.assert_called_once_with(
        model_size_or_path=MODEL_PATH_STT, # from clipExtractor
        device=DEVICE.type,
        compute_type="int8",
        local_files_only=False
    )
    assert clipExtractor_stt_model is mock_model_instance
    mock_logger.info.assert_any_call(f"✅ STT Model loaded successfully on [magenta]{DEVICE.type}[/magenta].")

def test_load_stt_model_failure(mocker, mock_logger):
    """Test STT model loading failure."""
    global clipExtractor_stt_model
    
    mock_whisper_constructor = mocker.patch('faster_whisper.WhisperModel')
    mock_whisper_constructor.side_effect = Exception("Test model loading error")

    load_stt_model()

    assert clipExtractor_stt_model is None
    mock_logger.error.assert_any_call(f"❌ ERROR loading STT model ({MODEL_PATH_STT}) on {DEVICE.type}: Test model loading error", exc_info=True)
    mock_logger.warning.assert_any_call("    Transcription will be simulated.")


def test_transcribe_video_with_mock_model(mocker, dummy_wav_file, mock_logger):
    """Test transcribe_video with a mocked STT model."""
    global clipExtractor_stt_model

    # Mock the global _stt_model used by transcribe_video
    mock_model = mocker.MagicMock()
    
    # Define the structure that WhisperModel.transcribe yields
    class MockSegmentInfo:
        def __init__(self, language, language_probability):
            self.language = language
            self.language_probability = language_probability

    class MockSegment:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text

    mock_segments_data = [
        MockSegment(0.0, 1.0, "Hello world."),
        MockSegment(1.0, 2.0, "This is a test.")
    ]
    mock_info = MockSegmentInfo("en", 0.99)

    mock_model.transcribe.return_value = (iter(mock_segments_data), mock_info)
    clipExtractor_stt_model = mock_model # Set the global model to our mock

    mocker.patch('subprocess.run') # Mock ffmpeg call for audio extraction

    result = transcribe_video(dummy_wav_file)

    expected_transcript = [
        {"start_time": 0.0, "end_time": 1.0, "text": "Hello world."},
        {"start_time": 1.0, "end_time": 2.0, "text": "This is a test."}
    ]
    assert result == expected_transcript
    mock_logger.info.assert_any_call(f"Detected language '[italic yellow]{mock_info.language}[/italic yellow]' with probability {mock_info.language_probability:.2f}")
    mock_model.transcribe.assert_called_once() # Check if the transcribe method was called

def test_transcribe_video_no_model(dummy_wav_file, mock_logger):
    """Test transcribe_video when STT model is None (simulated)."""
    global clipExtractor_stt_model
    clipExtractor_stt_model = None # Ensure model is None

    mocker.patch('subprocess.run') # Mock ffmpeg, though it might not be called if model is None early

    result = transcribe_video(dummy_wav_file)
    
    assert len(result) == 1
    assert result[0]['text'] == 'Simulated: STT model not available.'
    mock_logger.warning.assert_any_call(f"STT Model not loaded or failed to load. Simulating transcription for [cyan]{dummy_wav_file}[/cyan]")

def test_transcribe_video_ffmpeg_failure(mocker, dummy_wav_file, mock_logger):
    """Test transcribe_video when ffmpeg fails."""
    global clipExtractor_stt_model
    # Setup a mock STT model, so the function proceeds to ffmpeg call
    mock_model = mocker.MagicMock()
    clipExtractor_stt_model = mock_model

    # Mock subprocess.run to simulate ffmpeg failure
    mock_subprocess_run = mocker.patch('subprocess.run')
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="ffmpeg", stderr="ffmpeg error output"
    )

    result = transcribe_video(dummy_wav_file)

    assert len(result) == 1
    assert "Simulated: Error during STT." in result[0]['text'] # Or more specific error based on implementation
    mock_logger.error.assert_any_call(f"ERROR during actual STT transcription for [cyan]{dummy_wav_file}[/cyan]: CalledProcessError(\"Command 'ffmpeg' returned non-zero exit status 1.\")", exc_info=True)

# To run these tests, ensure scipy is installed for wavfile.write:
# pip install scipy pytest pytest-mock
# (faster_whisper would be needed if not mocking its constructor for real load tests)

# Note on WhisperModel constructor mock:
# If WhisperModel itself tries to do things in __init__ (like download files immediately if not found, even before transcribe),
# mocking its constructor (`mocker.patch('faster_whisper.WhisperModel', return_value=mock_model_instance)`)
# is crucial for hermetic unit tests. The `autospec=True` helps ensure the mock behaves like the original class.
# The current `load_stt_model` is already structured to handle exceptions during WhisperModel init.
# The global model `_stt_model` is what `transcribe_video` uses, so tests correctly assign to this via the fixture or direct patch.
