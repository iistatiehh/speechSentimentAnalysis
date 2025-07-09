import lightning as L
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Lightning version: {L.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")