import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json

# Load the labels for ImageNet classes
with open("imagenet_classes.json", "r") as f:
    labels = json.load(f)

    
# Define a function for image recognition
def recognize_image(image_path):
    # Load the image and apply the necessary transformations
    input_image = Image.open(image_path)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Check if a GPU is available and if not, use a CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    
    with torch.no_grad():
        input_batch = input_batch.to(device)
        output = model(input_batch)