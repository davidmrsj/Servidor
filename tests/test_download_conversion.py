import pytest
from unittest.mock import MagicMock, patch
import subprocess # Required for CompletedProcess

# Assuming clipExtractor.py is in app.services.services
# Adjust the import path based on your project structure and how pytest discovers modules
from app.services.services.clipExtractor import download_video, get_video_filename, log # Import log

# Mock the logger for testing log messages
@pytest.fixture
def mock_logger(mocker):
    return mocker.patch('app.services.services.clipExtractor.log')

def test_download_video_success(mocker, tmp_path, mock_logger):
    """Test successful video download simulation."""
    mock_subprocess_run = mocker.patch('subprocess.run')
    
    # Simulate successful yt-dlp execution
    # yt-dlp typically prints filename to stdout if --get-filename is used,
    # or just exits 0 on successful download.
    # For actual download, it would place a file.
    # We're mocking the part that would give us the filename.
    
    # Scenario 1: yt-dlp successfully downloads and we get the filename
    # (This is a simplified mock; actual yt-dlp interaction is more complex)
    expected_filename = "test_video.mp4"
    output_dir = tmp_path
    video_url = "https://www.youtube.com/watch?v=test"

    # Mock for the actual download command (simplified)
    mock_subprocess_run.return_value = MagicMock(
        returncode=0, 
        stdout=f"{expected_filename}\n".encode('utf-8'), # Simulate yt-dlp printing filename
        stderr=b""
    )
    
    # To simulate file creation by yt-dlp
    # We will create a dummy file that download_video should find
    downloaded_file_path = output_dir / expected_filename
    downloaded_file_path.touch() # Create the dummy file

    # Patch os.path.exists used within download_video to find the "downloaded" file
    mocker.patch('os.path.exists', return_value=True)
    # Patch os.path.join to ensure it works with tmp_path correctly if used internally by download_video for path construction
    mocker.patch('os.path.join', side_effect=lambda *args: output_dir.joinpath(*args[1:]))


    result_path = download_video(video_url, output_dir=str(output_dir))

    assert result_path == str(downloaded_file_path)
    mock_logger.info.assert_any_call(f"Video already exists: [green]{str(downloaded_file_path)}[/green]") # download_video checks existence first

    # More accurately, we should test the call to yt-dlp if the file doesn't exist first.
    # Let's refine this:
    mocker.resetall() # Reset mocks for a cleaner test case
    
    mock_subprocess_run = mocker.patch('subprocess.run')
    mocker.patch('os.path.exists', side_effect=[False, True]) # First False (not existing), then True (downloaded)
    
    mock_subprocess_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"") # Simulate download success
                                                                                        # yt-dlp download doesn't print filename to stdout by default
                                                                                        # it just creates the file.
                                                                                        # The filename is determined before calling yt-dlp.

    # The function `download_video` first tries to get the filename
    # then attempts download if file doesn't exist.
    
    # We need to mock the filename fetching part inside `download_video` if it's separate
    # For this test, assume `get_video_filename` is called internally or filename is predictable.
    # Let's assume `download_video` internally calls `get_video_filename`
    
    mocker.patch('app.services.services.clipExtractor.get_video_filename', return_value=expected_filename)


    # Simulate file creation by the mocked subprocess.run
    def side_effect_run(*args, **kwargs):
        # args[0] is the command list
        if "yt-dlp" in args[0][0] and "--get-filename" not in args[0]: # download command
            # Simulate file creation
            (output_dir / expected_filename).touch()
            return MagicMock(returncode=0, stdout=b"", stderr=b"")
        elif "--get-filename" in args[0]: # get_video_filename call (if not mocked separately)
             return MagicMock(returncode=0, stdout=f"{expected_filename}\n".encode('utf-8'), stderr=b"")
        return MagicMock(returncode=1, stdout=b"", stderr=b"Mocked error")

    mock_subprocess_run.side_effect = side_effect_run
    
    result_path = download_video(video_url, output_dir=str(output_dir))
    assert result_path == str(output_dir / expected_filename)
    mock_logger.info.assert_any_call(f"Successfully downloaded video to [green]{str(output_dir / expected_filename)}[/green]")


def test_download_video_failure(mocker, tmp_path, mock_logger):
    """Test failed video download simulation."""
    mock_subprocess_run = mocker.patch('subprocess.run')
    mocker.patch('app.services.services.clipExtractor.get_video_filename', return_value="test_video.mp4")
    mocker.patch('os.path.exists', return_value=False) # File does not exist

    # Simulate yt-dlp download failure
    mock_subprocess_run.return_value = MagicMock(
        returncode=1, 
        stdout=b"", 
        stderr=b"yt-dlp error"
    )

    video_url = "https://www.youtube.com/watch?v=testerror"
    output_dir = tmp_path
    
    result_path = download_video(video_url, output_dir=str(output_dir))

    assert result_path is None
    mock_logger.error.assert_any_call(f"Failed to download video from {video_url}. Error: yt-dlp error")

def test_get_video_filename_success(mocker, mock_logger):
    """Test successful filename retrieval."""
    mock_subprocess_run = mocker.patch('subprocess.run')
    expected_filename = "A YouTube Video Title.mp4"
    
    mock_subprocess_run.return_value = MagicMock(
        returncode=0,
        stdout=f"{expected_filename}\n".encode('utf-8'), # yt-dlp --get-filename output
        stderr=b""
    )
    
    video_url = "https://www.youtube.com/watch?v=some_id"
    filename = get_video_filename(video_url, "output_dir_placeholder") # output_dir not used by mock
    
    assert filename == expected_filename
    mock_logger.info.assert_any_call(f"Retrieved video filename: [cyan]{expected_filename}[/cyan]")

def test_get_video_filename_failure(mocker, mock_logger):
    """Test failed filename retrieval."""
    mock_subprocess_run = mocker.patch('subprocess.run')
    
    mock_subprocess_run.return_value = MagicMock(
        returncode=1,
        stdout=b"",
        stderr=b"yt-dlp error getting filename"
    )
    
    video_url = "https://www.youtube.com/watch?v=fail_id"
    filename = get_video_filename(video_url, "output_dir_placeholder")
    
    assert filename is None
    mock_logger.error.assert_any_call(f"Failed to get video filename for {video_url}. Error: yt-dlp error getting filename")

def test_download_video_already_exists(mocker, tmp_path, mock_logger):
    """Test scenario where video file already exists."""
    video_url = "https://www.youtube.com/watch?v=existing"
    expected_filename = "existing_video.mp4"
    output_dir = tmp_path
    existing_file_path = output_dir / expected_filename
    existing_file_path.touch() # Create the dummy "existing" file

    # Mock get_video_filename to return the name of the existing file
    mocker.patch('app.services.services.clipExtractor.get_video_filename', return_value=expected_filename)
    # Mock os.path.exists to confirm the file exists
    mocker.patch('os.path.exists', return_value=True)
    
    mock_subprocess_run = mocker.patch('subprocess.run') # Should not be called for download

    result_path = download_video(video_url, output_dir=str(output_dir))
    
    assert result_path == str(existing_file_path)
    mock_logger.info.assert_any_call(f"Video already exists: [green]{str(existing_file_path)}[/green]")
    mock_subprocess_run.assert_not_called() # Ensure yt-dlp download was not attempted

pytest_plugins = ['pytester'] # For pytester fixture, if needed for advanced tests, not used here.

# Note: The current `download_video` in `clipExtractor.py` has a slight logic issue.
# It calls `get_video_filename` which itself might create the "output_template" file via yt-dlp.
# Then it checks `os.path.exists(final_path)`. If `get_video_filename` already "downloaded" (e.g. using -o template),
# the main download step might be skipped due to file existence.
# The tests above try to cover this by mocking `get_video_filename` and `os.path.exists` carefully.
# A more robust `download_video` would separate filename fetching from actual downloading more cleanly.
# For now, tests are adapted to the current implementation.
