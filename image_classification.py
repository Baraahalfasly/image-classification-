import torch
from torchvision import models, transforms
from PIL import Image
import json
import os

# Load the pre-trained ResNet50 model from ImageNet
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()  # Set the model to evaluation mode

# Specify the image path and labels file path
img_path = 'images/image.png'  # Path to the image inside the 'images' folder
labels_path = 'files/imagenet_classes.json'  # Path to the labels inside the 'files' folder

# Check if the image exists at the given path
if not os.path.exists(img_path):
    print(f"The image does not exist at the provided path: {os.path.abspath(img_path)}")
else:
    # Load the image
    img = Image.open(img_path)

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

    # Classify the image
    with torch.no_grad():
        output = model(img_tensor)

    # Check if the labels file exists and load the class labels
    if not os.path.exists(labels_path):
        print(f"The labels file does not exist at the provided path: {os.path.abspath(labels_path)}")
    else:
        with open(labels_path) as f:
            labels = json.load(f)

        # Get the predicted class index and label
        _, predicted_idx = torch.max(output, 1)
        predicted_label = labels[str(predicted_idx.item())]

        # Print the result
        print(f"The image is classified as: {predicted_label}")
