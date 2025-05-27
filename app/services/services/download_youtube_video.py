import os
import subprocess # To call yt-dlp
import re
from typing import Optional

def get_video_filename(youtube_url: str, output_dir: str) -> Optional[str]:
    """
    Retrieves the expected filename for a YouTube video using yt-dlp's --get-filename.
    This helps check for existing files without downloading the whole video.
    Cleans the filename to be filesystem-friendly.
    """
    try:
        # Get the title or ID based filename yt-dlp would generate
        # -o specifies output template. %(title)s.%(ext)s or %(id)s.%(ext)s are common.
        # Using %(id)s for more stable filenames.
        cmd = [
            'yt-dlp',
            '--get-filename',
            '-o', '%(id)s.%(ext)s', # Output template: videoID.extension
            youtube_url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        raw_filename = result.stdout.strip()
        
        # Sanitize the filename (though yt-dlp often does this, an extra layer doesn't hurt)
        # Remove invalid characters, replace spaces with underscores
        sanitized_filename = re.sub(r'[\/*?:"<>|]', "", raw_filename)
        sanitized_filename = re.sub(r'\s+', '_', sanitized_filename)
        
        if not sanitized_filename:
            print(f"Error: Could not determine a valid filename for URL: {youtube_url}")
            return None
        return os.path.join(output_dir, sanitized_filename)

    except subprocess.CalledProcessError as e:
        print(f"Error getting filename with yt-dlp for {youtube_url}: {e}")
        print(f"yt-dlp stdout: {e.stdout}")
        print(f"yt-dlp stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while getting filename for {youtube_url}: {e}")
        return None

def download_video(youtube_url: str, output_dir: str = "app/media") -> Optional[str]:
    """
    Downloads a video from the given YouTube URL using yt-dlp.

    Args:
        youtube_url: The URL of the YouTube video.
        output_dir: The directory to save the downloaded video. 
                    Defaults to "app/media".

    Returns:
        The full path to the downloaded video file if successful, 
        otherwise None.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine the expected output filename
    # This uses yt-dlp to get the filename without downloading if possible,
    # or constructs a generic one if that fails.
    # Using video ID as filename for consistency and to avoid special characters.
    expected_filename_path = get_video_filename(youtube_url, output_dir)

    if not expected_filename_path:
        print(f"Could not determine filename for {youtube_url}. Download aborted.")
        return None

    # Check if the video already exists
    if os.path.exists(expected_filename_path):
        print(f"Video already exists: {expected_filename_path}")
        return expected_filename_path

    print(f"Attempting to download video: {youtube_url} to {expected_filename_path}...")
    try:
        # Using -o to specify the exact output path and filename
        # Format selection: best video and audio, merged. mp4 preferred.
        # Example: -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
        cmd = [
            'yt-dlp',
            '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', # Standard format selection
            '-o', expected_filename_path, # Output template
            '--merge-output-format', 'mp4', # Ensure final output is mp4 if merging
            youtube_url
        ]
        
        # Run yt-dlp
        # Increased timeout for potentially long downloads.
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600) # 1 hour timeout

        print(f"Video downloaded successfully: {expected_filename_path}")
        return expected_filename_path

    except subprocess.CalledProcessError as e:
        print(f"Error downloading video with yt-dlp: {youtube_url}")
        print(f"Return code: {e.returncode}")
        print(f"Output (stdout): {e.stdout}")
        print(f"Output (stderr): {e.stderr}")
        # Try to remove partially downloaded file if error occurred
        if os.path.exists(expected_filename_path):
            try:
                os.remove(expected_filename_path)
                print(f"Removed partially downloaded file: {expected_filename_path}")
            except OSError as oe:
                print(f"Error removing partially downloaded file {expected_filename_path}: {oe}")
        return None
    except subprocess.TimeoutExpired:
        print(f"Timeout occurred while downloading {youtube_url}. Download may be incomplete.")
        # Try to remove partially downloaded file
        if os.path.exists(expected_filename_path):
            try:
                os.remove(expected_filename_path)
                print(f"Removed partially downloaded file due to timeout: {expected_filename_path}")
            except OSError as oe:
                print(f"Error removing partially downloaded file {expected_filename_path} after timeout: {oe}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == '__main__':
    # Example Usage (for testing the module directly)
    test_url_short = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Short video
    # A slightly longer video example - find a copyright-free one if possible for actual testing
    test_url_long = "https://www.youtube.com/watch?v=GGlYgH9471o" # Example: Blender Open Movie

    print(f"--- Testing with short video: {test_url_short} ---")
    file_path_short = download_video(test_url_short, output_dir="app/media_test_download")
    if file_path_short:
        print(f"Downloaded to: {file_path_short}")
        print(f"File exists: {os.path.exists(file_path_short)}")
    else:
        print("Download failed for short video.")

    # print(f"\n--- Testing with potentially longer video: {test_url_long} ---")
    # file_path_long = download_video(test_url_long, output_dir="app/media_test_download")
    # if file_path_long:
    #     print(f"Downloaded to: {file_path_long}")
    #     print(f"File exists: {os.path.exists(file_path_long)}")
    # else:
    #     print("Download failed for long video.")
    
    # Test re-download attempt
    print(f"\n--- Testing re-download for short video: {test_url_short} ---")
    file_path_short_again = download_video(test_url_short, output_dir="app/media_test_download")
    if file_path_short_again:
        print(f"Re-download attempt path: {file_path_short_again}")
    else:
        print("Re-download failed (or was skipped as expected).")
