from io import BytesIO
from app.config import DEVICE, MODEL_PATH_IMAGE_SCALABLE
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
import os
import numpy as np
from PIL import Image
from gfpgan import GFPGANer

class ImageUpscaler:

    def __init__(self, model_path: str = MODEL_PATH_IMAGE_SCALABLE, device=DEVICE):
        print(f"Inicializando ImageUpscaler con dispositivo: {device}")

        self.device = device
        self.model_path = model_path

        # Definir el modelo RRDBNet para Real-ESRGAN
        rrdbnet = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4
        ).to(self.device) 

        # Crear el RealESRGANer
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=self.model_path,
            model=rrdbnet,
            tile=256,       # tamaño del tile
            tile_pad=10,    # relleno para evitar bordes
            pre_pad=10,
            half=True,
            device=self.device
        )

        print("Modelo Real-ESRGAN cargado exitosamente.")

    def upscale_image_bytes(self, image_bytes: bytes) -> bytes:
        """
        Recibe una imagen en formato bytes, la escala a 4K y devuelve la imagen resultante en bytes (formato PNG).
        """
        # Cargar la imagen desde bytes y convertirla a RGB
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(img)  # Mantén la imagen en formato NumPy

        print(f"Imagen convertida a NumPy, shape: {image_np.shape}")

        # Realizar la mejora de resolución (asegurar que es NumPy)
        with torch.no_grad():
            output, _ = self.upsampler.enhance(image_np, outscale=4) 

        # Convertir el resultado a bytes en formato PNG
        out_img = Image.fromarray(output.astype(np.uint8))  # Convertir de nuevo a uint8
        out_buffer = BytesIO()
        out_img.save(out_buffer, format="PNG")
        out_buffer.seek(0)
        
        return out_buffer.getvalue()
