# Proyecto FastAPI + GFPGAN + Real-ESRGAN

Este repositorio contiene un servidor FastAPI que ofrece dos endpoints de super-resolución y restauración de caras:

- **`/escalar-imagen/`**: Escala cualquier imagen 4× usando Real-ESRGAN.  
- **`/restaurar-cara/`**: Restaura y aumenta resolución de rostros con GFPGAN.

Además incluye un ejemplo de integración en Android (Retrofit + Hilt).

---

## 📋 Índice

1. [Requisitos](#-requisitos)  
2. [Estructura de carpetas clave](#-estructura-de-carpetas-clave)  
3. [Instalación y configuración (backend)](#-instalación-y-configuración-backend)  
4. [Descarga y ubicación de modelos](#-descarga-y-ubicación-de-modelos)  
5. [Arrancar el servidor FastAPI](#-arrancar-el-servidor-fastapi)  
6. [Configurar y compilar la app Android](#-configurar-y-compilar-la-app-android)  
7. [Pruebas con cURL](#-pruebas-con-curl)  

---

## 🔧 Requisitos

- **Python 3.7+** (recomendado 3.10 o 3.11)  
- **pip**  
- **NVIDIA GPU + CUDA/cuDNN**  
- **Node/NPM**
- **Git**  

---

## 📂 Estructura de carpetas clave

```text
project-root/
├── app/                      ← Código FastAPI
│   ├── main.py
│   ├── config.py
│   └── services/
│       ├── upscaler_service.py
│       └── gfpgan_service.py
├── models/              ← Carpeta local no versionada
│   ├── RealESRGAN_x4plus.pth
│   └── GFPGANv1.3.pth
├── gfpgan/
│   └── weights/              ← Modelos de detección de facexlib
│       ├── detection_Resnet50_Final.pth
│       └── parsing_parsenet.pth
├── requirements.txt
├── .gitignore
└── README.md
```
## 3. Instalación y configuración (backend)

```bash
# 1. Clona el repositorio
git clone https://github.com/TU_USUARIO/fastapi-gfpgan-server.git
cd fastapi-gfpgan-server

# 2. Crea y activa un entorno virtual
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 3. Instala dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Descarga y ubicación de modelos

> **Nota:** estos archivos **NO** están incluidos en el repo. Tendras que crear en el raiz del proyecto la carpeta models

- **Real-ESRGAN x4**  
  1. Descarga `RealESRGAN_x4plus.pth` desde  
     https://github.com/xinntao/Real-ESRGAN/releases  
  2. Copia a `models/RealESRGAN_x4plus.pth`

- **GFPGAN v1.3**  
  1. Descarga `GFPGANv1.3.pth` desde  
     https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth  
  2. Copia a `models/GFPGANv1.3.pth` 

## 5️⃣ Arrancar el servidor FastAPI

```bash
# Activa el entorno si no está activo
venv\Scripts\activate    # Windows
# o
source venv/bin/activate  # Linux/macOS

# Lanza el servidor FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Ahora en la aplicacion android deberemos introducir la ip de su ordenador para que la aplicacion pueda enviar peticiones al servidor que sera nuestro ordenador donde haya ejecutado el proyecto por ejemplo: 192.168.1.136

Y ya tendria configurado todo y listo para escalar tus fotos
