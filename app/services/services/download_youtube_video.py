import os
import subprocess # To call yt-dlp
import re
from typing import Optional
from pathlib import Path

COOKIES_FILE = r"C:\Servidor\cookies.txt" 

def get_video_filename(youtube_url: str, output_dir: str) -> Optional[str]:
    """
    Devuelve el nombre de archivo que yt-dlp generaría para un vídeo,
    usando las cookies para saltar los chequeos de bot.
    """
    try:
        cmd = [
            "yt-dlp",
            "--cookies", COOKIES_FILE,
            "--get-filename",
            "-o", "%(id)s.%(ext)s",           # siempre <ID>.ext
            youtube_url,
        ]
        result = subprocess.run(
            cmd, text=True, capture_output=True, check=True, encoding="utf-8"
        )
        raw_name = result.stdout.strip()

        # limpieza mínima extra
        clean_name = re.sub(r'[\/*?:"<>|]', "", raw_name)
        clean_name = re.sub(r"\s+", "_", clean_name)

        if not clean_name:
            print(f"No se pudo determinar nombre para {youtube_url}")
            return None
        return os.path.join(output_dir, clean_name)

    except subprocess.CalledProcessError as e:
        print("yt-dlp --get-filename falló:", e.stderr or e.stdout)
        return None
    except Exception as e:
        print("Error inesperado en get_video_filename:", e)
        return None

def download_video(youtube_url: str, output_dir: str = "app/media") -> Optional[str]:
    """
    Descarga el vídeo (mp4) usando las mismas cookies.
    Si el archivo ya existe, lo reutiliza.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "%(id)s.%(ext)s")

    # --- consultar ID para ver si ya está ---
    try:
        vid_id = subprocess.check_output(
            ["yt-dlp", "--cookies", COOKIES_FILE, "--print", "id", "-q", youtube_url],
            text=True
        ).strip()
        target = out_dir / f"{vid_id}.mp4"
        if target.exists():
            print(f"✔️  Ya estaba descargado: {target}")
            return str(target)
    except Exception:
        target = None  # si falla, descargará igualmente

    # --- construir comando de descarga (con cookies) -------------
    ydl_cmd = [
        "yt-dlp",
        "--cookies", COOKIES_FILE,
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "--merge-output-format", "mp4",
        "-o", out_template,
        youtube_url,
    ]

    # --- descargar ------------------------------------------------
    try:
        print("⬇️  Descargando con yt-dlp…")
        subprocess.run(ydl_cmd, check=True)
    except subprocess.CalledProcessError as err:
        print("❌  yt-dlp devolvió error:", err)
        if target and target.exists():
            target.unlink(missing_ok=True)
        return None

    # --- localizar archivo final ---------------------------------
    if not target:
        mp4s = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        target = mp4s[0] if mp4s else None

    if target and target.exists():
        print(f"✅  Descargado: {target}")
        return str(target)

    print("❌  No se encontró el archivo descargado.")
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
