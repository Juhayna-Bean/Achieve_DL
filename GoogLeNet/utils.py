import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn as nn

def get_demo_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.5)
    ])

    test_loader = DataLoader(CIFAR10('./data', train=False, download=True, transform=transform), batch_size=64, shuffle=False)
    return next(iter(test_loader))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model, device="cpu", as_string=True):
    model.to(device)
    size_bytes = torch.cuda.memory_allocated() if device == "cuda" else 0
    size_bytes += sum(p.numel() * p.element_size() for p in model.parameters())
    size_bytes += sum(p.numel() * p.element_size() for p in model.buffers())
    size_bytes += sum(p.numel() * p.element_size() for p in model.children() if isinstance(p, nn.AdaptiveAvgPool2d))
    size_mb = size_bytes / (1024 * 1024)
    if as_string:
        return f"{size_mb:.2f} Mb"
    else:
        return size_mb


def _part(data, n):
    i = 0
    data = iter(data)
    while i<n:
        yield next(data)
        i += 1

class part(): 
    def __init__(self, data, n) -> None:
        self.data = data
        self.n = n
    
    def __iter__(self):
        return _part(self.data,self.n)
    
    def __len__(self):
        return self.n
