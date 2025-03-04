import torch
print(torch.cuda.is_available())  # 如果返回 False，说明 PyTorch 不支持 CUDA
print(torch.version.cuda)         # 查看 CUDA 版本