import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Get the directory of the current script (main.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two levels to reach the project root, then into the models directory
model_dir = os.path.normpath(os.path.join(current_dir, "../../models"))

dataset_dir = os.path.normpath(os.path.join(current_dir, "../../dataset"))

train_dir = os.path.join(dataset_dir, "classification train species" ,"train")


model_paths = {
    #"ViT": os.path.join(model_dir, "vit_classification_model.pth"),
    "EfficientNetV2": os.path.join(model_dir, "efficientnet_best.pth"),
    "MobileNet": os.path.join(model_dir, "mobilenet_best.pth"),
    "RegNet": os.path.join(model_dir, "regnet_best.pth"),
    "ResNet": os.path.join(model_dir, "resnet_best.pth"),
}
# Define the augmentations and transformations
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),  # Randomly crop and resize the image
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(20),  # Random rotation
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        ),  # Random color jitter
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
        transforms.RandomErasing(),  # Random erasing for regularization
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

transform_val = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# Load the dataset
full_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)

# Split into train and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size]
)

val_dataset.dataset.transform = transform_val

val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, data_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    return cm, report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Blood Oranges_transform', 'Navel_transform', 'Tangelo_transform', 'Tangerine_transform', 'cara cara_transform']  # Replace with your actual class names

from torchvision.models import regnet_y_400mf  # Example model

# Step 1: Define the RegNet model
class RegNetModel(nn.Module):
    def __init__(self, num_classes=5):
        super(RegNetModel, self).__init__()
        # Load pre-trained RegNet-Y 400MF model
        self.model = regnet_y_400mf(pretrained=True)

        # Modify the final layer to match the number of classes in your dataset
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
num_classes = 5

model = RegNetModel(num_classes=num_classes) # Initialize model architecture
model.load_state_dict(torch.load(model_paths["RegNet"]))  # Load the saved weights
model.eval()
model = model.to(device)


# Evaluate the model
confusion_mat, class_report = evaluate_model(model, val_loader, device, class_names)

# Print results
print("Confusion Matrix:")
print(confusion_mat)
print("\nClassification Report:")
print(class_report)