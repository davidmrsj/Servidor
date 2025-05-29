# main.py
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, StreamingResponse
from app.services.services.upscalerImages import ImageUpscaler
from app.services.services.qualityFaceConversion import GFPGANService
import clipExtractor as clipExtractor

app = FastAPI()

# Cargamos la clase en una variable global (para que no se recargue en cada request)
upscaler = ImageUpscaler()
qualityFaces = GFPGANService()

clipExtractor.main()

@app.get("/")
async def home():
    return {"status": "Servidor activo y funcionando con GPU"}

@app.post("/escalar-imagen/")
async def escalar_imagen(file: UploadFile = File(...)):
    # Leer los bytes de la imagen que subió el usuario
    img_bytes = await file.read()
    
    scaled_bytes = qualityFaces.restore_face(img_bytes)
    # Escalar la imagen a 4K
    #scaled_bytes = upscaler.upscale_image_bytes(faceScale)

    # Devolver la imagen resultante con el tipo MIME adecuado
    # Nota: Usamos 'image/png' porque en upscale_image_bytes usamos formato PNG.
    return StreamingResponse(BytesIO(scaled_bytes), media_type="image/png")
