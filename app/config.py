# app/config.py

import torch
import os

# Dispositivo: cuda si está disponible, sino cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ruta al modelo RealESRGAN
# Puedes ajustarlo según tu estructura
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_IMAGE_SCALABLE = os.path.join(os.path.dirname(BASE_DIR), "models", "RealESRGAN_x4plus.pth")
MODEL_PATH_FACES = os.path.join(os.path.dirname(BASE_DIR), "models", "GFPGANv1.3.pth")

# Directorios de entrada / salida (si lo usas de forma fija)
INPUT_IMAGE_DIR = os.path.join(os.path.dirname(BASE_DIR), "input_images")
OUTPUT_IMAGE_DIR = os.path.join(os.path.dirname(BASE_DIR), "output_images")
