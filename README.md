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
- **Microsoft visual 2015-2022**
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
