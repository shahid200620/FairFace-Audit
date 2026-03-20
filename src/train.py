import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import random

from model import FaceModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.ToTensor()
])

path = "data/UTKFace"
files = os.listdir(path)

def get_pair():
    img1 = random.choice(files)
    img2 = random.choice(files)

    label = 1 if img1.split("_")[0] == img2.split("_")[0] else 0

    i1 = Image.open(os.path.join(path, img1)).convert("RGB")
    i2 = Image.open(os.path.join(path, img2)).convert("RGB")

    return transform(i1), transform(i2), label

model = FaceModel().to(device)
criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):
    total_loss = 0
    for _ in range(200):
        x1, x2, label = get_pair()

        x1 = x1.unsqueeze(0).to(device)
        x2 = x2.unsqueeze(0).to(device)

        label = torch.tensor([1 if label == 1 else -1]).to(device)

        out1 = model(x1)
        out2 = model(x2)

        loss = criterion(out1, out2, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("epoch:", epoch, "loss:", total_loss)

torch.save(model.state_dict(), "artifacts/model.pt")