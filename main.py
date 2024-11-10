import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json

# Load the labels for ImageNet classes
with open("imagenet_classes.json", "r") as f:
    labels = json.load(f)
