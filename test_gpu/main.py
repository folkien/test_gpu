import torch
import torchvision

def check_cuda_support():
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"CUDA is available. CUDA version: {cuda_version}")
    else:
        print("CUDA is not available.")

def check_cpu_support():
    try:
        # Attempt to create a tensor on CPU
        tensor = torch.randn(1)
        print("CPU is supported.")
    except Exception as e:
        print(f"CPU is not supported. Error: {e}")

def check_torchvision_support():
    try:
        # Attempt to access a torchvision function
        model = torchvision.models.resnet18(pretrained=False)
        print("torchvision is supported.")
    except Exception as e:
        print(f"torchvision is not supported. Error: {e}")

def check_torchvision_cuda_support():
    if torch.cuda.is_available():
        try:
            # Attempt to create a model and move it to GPU
            model = torchvision.models.resnet18(pretrained=False).cuda()
            print("torchvision supports CUDA.")
        except Exception as e:
            print(f"torchvision does not support CUDA. Error: {e}")
    else:
        print("CUDA is not available, skipping torchvision CUDA support check.")

def main():
    print("Checking PyTorch and torchvision support...\n")
    
    # Check PyTorch support
    print(f"PyTorch version: {torch.__version__}")
    check_cpu_support()
    check_cuda_support()

    # Check torchvision support
    print(f"\ntorchvision version: {torchvision.__version__}")
    check_torchvision_support()
    check_torchvision_cuda_support()

if __name__ == "__main__":
    main()

