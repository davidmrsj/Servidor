import torch
print(torch.__version__)  # Verifica la versión de PyTorch
print(torch.version.cuda)  # Verifica la versión de CUDA en PyTorch
print(torch.backends.cudnn.version())  # Verifica si CuDNN está disponible

