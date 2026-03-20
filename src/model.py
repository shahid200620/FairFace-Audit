import torch
import torch.nn as nn
import torchvision.models as models

class FaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, 128)

    def forward(self, x):
        return self.base(x)