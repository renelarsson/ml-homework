#!/usr/bin/env python3
import json
from io import BytesIO
from urllib import request
from PIL import Image
from torchvision import transforms

# Step 1: Download the image
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

# Step 2: Prepare the image
def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# Step 3: Preprocess the image
def preprocess_image(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform(img)

# Download and preprocess the image
url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
img = download_image(url)
img = prepare_image(img, (200, 200))
x_tensor = preprocess_image(img)

# Convert to JSON format
x_input = x_tensor.unsqueeze(0).numpy().tolist()
json_data = {"image": x_input}

# Save JSON data to a file
with open("json_data.json", "w") as f:
    json.dump(json_data, f)