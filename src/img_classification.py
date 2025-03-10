"""
Date: 03.March.2025
Author: Harish Natarajan Ravi
Email: harrish.nr@gmail.com
"""

import subprocess
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torchsummary import summary

# Ensure 'opendatasets' is installed
try:
    import opendatasets as od
except ImportError:
    subprocess.check_call(["pip", "install", "opendatasets"])
    import opendatasets as od

# Download dataset
DATASET_URL = "https://www.kaggle.com/datasets/andrewmvd/animal-faces"
od.download(DATASET_URL)
print("Dataset downloaded successfully!")

# Device Configuration (Detect GPU if available, otherwise use CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device available:", device)

# Load Image Paths and Labels
image_paths, labels = [], []
DATASET_DIR = "./animal-faces/afhq/"

for category in os.listdir(DATASET_DIR):
    for label in os.listdir(os.path.join(DATASET_DIR, category)):
        for img in os.listdir(os.path.join(DATASET_DIR, category, label)):
            labels.append(label)
            image_paths.append(os.path.join(DATASET_DIR, category, label, img))

# Create a DataFrame
data_df = pd.DataFrame(zip(image_paths, labels), columns=["image_paths", "labels"])
print(data_df.head())

# Splitting the Dataset
train_df = data_df.sample(frac=0.7, random_state=7)
test_df = data_df.drop(train_df.index)
val_df = test_df.sample(frac=0.5, random_state=7)
test_df = test_df.drop(val_df.index)

# Encode Labels
label_encoder = LabelEncoder()
label_encoder.fit(data_df["labels"])

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.tensor(label_encoder.transform(dataframe["labels"])).to(device)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image).to(device)
        return image, label
    
# Create Dataset Instances
train_dataset = CustomImageDataset(train_df, transform)
val_dataset = CustomImageDataset(val_df, transform)
test_dataset = CustomImageDataset(test_df, transform)


# Visualizing Sample Images
n_rows, n_cols = 3, 3
fig, axarr = plt.subplots(n_rows, n_cols)
for row in range(n_rows):
    for col in range(n_cols):
        img_path = data_df.sample(n=1)["image_paths"].iloc[0]
        img = Image.open(img_path).convert("RGB")
        axarr[row, col].imshow(img)
        axarr[row, col].axis("off")
plt.savefig("image_grid.png")
print("Image grid saved as 'image_grid.png'")

# Hyperparameters
LR = 1e-4
BATCH_SIZE = 16
EPOCHS = 10

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# CNN Model Definition
class Net(nn.Module):
    def __init__(self):
      super().__init__()

      self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1) # First Convolution layer
      self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1) # Second Convolution layer
      self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1) # Third Convolution layer
      self.pooling = nn.MaxPool2d(2,2) # The pooling layer, we will be using the same layer after each conv2d.
      self.relu = nn.ReLU() # ReLU Activation function

      self.flatten = nn.Flatten() # Flatten and vectorize the output feature maps that somes from the final convolution layer.
      self.linear = nn.Linear((128 * 16 * 16), 128) # Traditional Dense (Linear)
      self.output = nn.Linear(128, len(data_df['labels'].unique())) # Output Linear Layer


    def forward(self, x):
      x = self.conv1(x) # -> Outputs: (32, 128, 128)
      x = self.pooling(x)# -> Outputs: (32, 64, 64)
      x = self.relu(x)
      x = self.conv2(x) # -> Outputs: (64, 64, 64)
      x = self.pooling(x) # -> Outputs: (64, 32, 32)
      x = self.relu(x)
      x = self.conv3(x) # -> Outputs: (128, 32, 32)
      x = self.pooling(x) # -> Outputs: (128, 16, 16)
      x = self.relu(x)
      x = self.flatten(x)
      x = self.linear(x)
      x = self.output(x)

      return x


# Model Initialization
model = Net().to(device)
summary(model, input_size=(3, 128, 128))

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

# Training
total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []


for epoch in range(EPOCHS):
  total_acc_train = 0
  total_loss_train = 0
  total_loss_val = 0
  total_acc_val = 0

  for inputs, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    train_loss = criterion(outputs, labels)
    total_loss_train += train_loss.item()
    train_loss.backward()

    train_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
    total_acc_train += train_acc
    optimizer.step()

  with torch.no_grad():
    for inputs, labels in val_loader:
      outputs = model(inputs)
      val_loss = criterion(outputs, labels)
      total_loss_val += val_loss.item()

      val_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
      total_acc_val += val_acc

  total_loss_train_plot.append(round(total_loss_train/1000, 4))
  total_loss_validation_plot.append(round(total_loss_val/1000, 4))
  total_acc_train_plot.append(round(total_acc_train/(train_dataset.__len__())*100, 4))
  total_acc_validation_plot.append(round(total_acc_val/(val_dataset.__len__())*100, 4))
  print(f'''Epoch {epoch+1}/{EPOCHS}, Train Loss: {round(total_loss_train/100, 4)} Train Accuracy {round((total_acc_train)/train_dataset.__len__() * 100, 4)}
              Validation Loss: {round(total_loss_val/100, 4)} Validation Accuracy: {round((total_acc_val)/val_dataset.__len__() * 100, 4)}''')
  print("="*25)

# Testing
with torch.no_grad():
  total_loss_test = 0
  total_acc_test = 0
  for inputs, labels in test_loader:
    predictions = model(inputs)

    acc = (torch.argmax(predictions, axis = 1) == labels).sum().item()
    total_acc_test += acc
    test_loss = criterion(predictions, labels)
    total_loss_test += test_loss.item()

print(f"Accuracy Score is: {round((total_acc_test/test_dataset.__len__()) * 100, 4)} and Loss is {round(total_loss_test/1000, 4)}")

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
axs[0].plot(total_loss_train_plot, label='Training Loss')
axs[0].plot(total_loss_validation_plot, label='Validation Loss')
axs[0].set_title('Training and Validation Loss over Epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(total_acc_train_plot, label='Training Accuracy')
axs[1].plot(total_acc_validation_plot, label='Validation Accuracy')
axs[1].set_title('Training and Validation Accuracy over Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

plt.tight_layout()

# plt.show()
plt.savefig("training_progress.png")
print("Plot image saved as 'training_progress.png'") 

# Inference

# 1- read image
# 2- Transform using transform object
# 3- predict through the model
# 4- inverse transform by label encoder

def predict_image(image_path):
  image = Image.open(image_path).convert('RGB')
  image = transform(image).to(device)

  output = model(image.unsqueeze(0))
  output = torch.argmax(output, axis = 1).item()
  return label_encoder.inverse_transform([output])

## Visualize the image
inf_img1 = Image.open("./animal-faces/inference/cute-photos-of-cats-looking-at-camera-1593184780.jpg")

plt.figure(figsize=(4, 4))
plt.imshow(inf_img1)
plt.title("Inf Image1")

plt.savefig("inf_img1.png")
print("Inference image saved as 'inf_img1.png'")

inf_img2 = Image.open("./animal-faces/inference/photos_lion_king.jpg")

plt.figure(figsize=(4, 4))
plt.imshow(inf_img2)
plt.title("Inf Image2")

plt.savefig("inf_img2.png")
print("Inference image saved as 'inf_img2.png'") 
# plt.show()


## Predict
print()
print("Prediction image1: \n")
predict_output1 = predict_image("./animal-faces/inference/cute-photos-of-cats-looking-at-camera-1593184780.jpg")
print(predict_output1)
print("\n")
print("Prediction image2: \n")
predict_output2 = predict_image("./animal-faces/inference/photos_lion_king.jpg")
print(predict_output2)
print("\n")