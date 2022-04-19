import torch
import torchsuite
import pytorchcheckpoint
from deepgesture.config import Config, check_paths

if __name__ == "__main__":
    print("installation working")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Checking paths")
    check_paths()
