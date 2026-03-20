import torch
import os
import random
import json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from model import FaceModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.ToTensor()
])

path = "data/UTKFace"
files = os.listdir(path)

model = FaceModel().to(device)
model.load_state_dict(torch.load("artifacts/model.pt", map_location=device))
model.eval()

def get_info(name):
    parts = name.split("_")
    age = int(parts[0])
    gender = "Male" if parts[1] == "0" else "Female"
    race = int(parts[2])
    return age, gender, race

def age_group(age):
    if age <= 19:
        return "0-19"
    elif age <= 39:
        return "20-39"
    elif age <= 59:
        return "40-59"
    else:
        return "60+"

def tone_group(r):
    if r == 0:
        return "Light"
    elif r in [1,2]:
        return "Medium"
    else:
        return "Dark"

def get_pair():
    f1 = random.choice(files)
    f2 = random.choice(files)

    same = f1.split("_")[0] == f2.split("_")[0]

    i1 = Image.open(os.path.join(path, f1)).convert("RGB")
    i2 = Image.open(os.path.join(path, f2)).convert("RGB")

    return f1, f2, transform(i1), transform(i2), same

threshold = 0.5

results = {}

tp = fp = tn = fn = 0

for _ in range(500):
    f1, f2, x1, x2, same = get_pair()

    x1 = x1.unsqueeze(0).to(device)
    x2 = x2.unsqueeze(0).to(device)

    with torch.no_grad():
        e1 = model(x1)
        e2 = model(x2)

    sim = F.cosine_similarity(e1, e2).item()

    pred = sim > threshold

    if same and pred:
        tp += 1
    elif not same and pred:
        fp += 1
    elif not same and not pred:
        tn += 1
    else:
        fn += 1

far = fp / (fp + tn + 1e-6)
frr = fn / (fn + tp + 1e-6)

results["overall"] = {
    "far": float(far),
    "frr": float(frr)
}

with open("results/initial_audit.json", "w") as f:
    json.dump(results, f, indent=2)

print("done")