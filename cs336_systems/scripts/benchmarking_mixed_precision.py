import torch
from torch import nn
import torch.nn.functional as F

from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_image, secrets


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("input:", x.dtype)
        x = self.relu(self.fc1(x))
        print("output of first layer:", x.dtype)
        x = self.ln(x)
        print("output of layer norm:", x.dtype)
        x = self.fc2(x)
        print("predicted logits:", x.dtype)
        return x


def print_dtypes():
    model = ToyModel(3, 2).to("cuda")
    
    x = torch.randn(1, 3, dtype=torch.float32, device="cuda")
    y = torch.randint(2, (1,), device="cuda")
    
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        print("model.fc1.weight.dtype:", model.fc1.weight.dtype)
        print("model.fc2.weight.dtype:", model.fc2.weight.dtype)

        preds = model(x)

        loss = F.cross_entropy(preds, y)
        print("loss:", loss.dtype)
    
    loss.backward()
    print("model.fc1.weight.grad.dtype:", model.fc1.weight.grad.dtype)
    print("model.fc2.weight.grad.dtype:", model.fc2.weight.grad.dtype)


@app.function(image=build_image(), secrets=secrets(), volumes=VOLUME_MOUNTS, gpu="B200", timeout=45*60)
def print_dtypes_modal():
    return print_dtypes()
