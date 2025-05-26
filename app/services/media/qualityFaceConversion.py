from io import BytesIO
from app.config import MODEL_PATH_FACES, DEVICE
import torch
import numpy as np
from PIL import Image
from gfpgan import GFPGANer

class GFPGANService:
    def __init__(self, model_path= MODEL_PATH_FACES, device=DEVICE):
        print(f"Inicializando GFPGAN en: {device}")

        self.device = device
        self.model_path = model_path

        # Cargar modelo GFPGAN
        self.gfpgan = GFPGANer(
            model_path=self.model_path,
            upscale=2,  
            arch='clean',  
            channel_multiplier=2,
            bg_upsampler=None,
            device=self.device
        )

    def restore_face(self, image_bytes: bytes) -> bytes:
        """ Recibe una imagen en bytes, la restaura con GFPGAN y devuelve la imagen restaurada en bytes """
        
        # Convertir la imagen de bytes a PIL
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(img)

        print(f"Procesando imagen con forma: {image_np.shape}")

        # Restaurar la imagen
        _, _, restored_img = self.gfpgan.enhance(image_np, has_aligned=False, only_center_face=False, paste_back=True)

        if restored_img is None:
            raise ValueError("Error en GFPGAN: La imagen restaurada es None.")

        print(f"Dimensiones de la imagen restaurada: {restored_img.shape}")
        # Convertir de nuevo a bytes
        restored_pil = Image.fromarray(restored_img.astype(np.uint8))
        output_buffer = BytesIO()
        restored_pil.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        return output_buffer.getvalue()
