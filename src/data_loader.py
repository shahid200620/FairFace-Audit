import os

path = "data/UTKFace"

files = os.listdir(path)

print("total images:", len(files))

if len(files) > 0:
    print("sample file:", files[0])