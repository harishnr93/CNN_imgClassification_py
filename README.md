# Image Classification using Convolutional Neural Network (CNN)

## Overview
This project implements a deep learning pipeline to classify images of animal faces and bean-leaf-lesions using a Convolutional Neural Network (CNN) and pretrained model. The dataset is downloaded from Kaggle and consists of images categorized into different classes.

## Features
- Automatically downloads and prepares the dataset from Kaggle.
- Preprocesses images using transformations.
- Implements a custom dataset class for loading images.
- Builds a CNN model using PyTorch.
- Trains the model with accuracy and loss tracking.
- Evaluates the trained model on validation and test datasets.
- Provides inference on new images.

## Requirements
Ensure you have the following installed:
- Python 3.x
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- opendatasets
- scikit-learn
- PIL (Pillow)
- torchsummary

You can install the required packages using:
```bash
pip install torch torchvision numpy pandas matplotlib opendatasets scikit-learn pillow torchsummary
```

## Dataset
The dataset is automatically downloaded from Kaggle. If you haven't set up Kaggle API access, ensure you have an API key stored in `~/.kaggle/kaggle.json`.

Dataset URL: [Animal Faces Dataset](https://www.kaggle.com/datasets/andrewmvd/animal-faces)
Dataset URL: [Bean Leaf Lesions Dataset](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification)

## Steps
### 1. Data Preparation
- Downloads the dataset and extracts images.
- Reads image paths and labels.
- Splits data into training (70%), validation (15%), and test (15%) sets.
- Encodes labels using `LabelEncoder`.
- Defines image transformations (resize to 128x128, convert to tensor, normalize).

### 2. Custom Dataset Class
A custom dataset class `CustomImageDataset` is implemented to:
- Load images from disk.
- Apply transformations.
- Convert labels to tensor format.

### 3. Model Architecture
The CNN model consists of:
- Three convolutional layers with ReLU activation and max pooling.
- A flattening layer followed by fully connected layers.
- An output layer with softmax activation (implemented as logits in PyTorch).

### 4. Training Process
- Uses cross-entropy loss.
- Adam optimizer with a learning rate of `1e-4`.
- Trains for 10 epochs.
- Tracks and plots training and validation accuracy/loss.

### 5. Evaluation
- Computes test accuracy and loss.
- Visualizes training progress using plots.

### 6. Inference
- Loads an image for testing.
- Transforms it using the same preprocessing steps.
- Passes it through the trained model.
- Decodes the predicted label using `LabelEncoder`.

## Results
- Accuracy and loss graphs are saved as `training_progress.png`.
- Sample inference images are saved as `inf_img1.png` and `inf_img2.png`.
- Predictions for new images are displayed in the console.

## Running the Project
To execute the pipeline, simply run:
```bash
python3 ./img_classification.py
```
Ensure the dataset is downloaded and stored at the correct path before training begins.

## Future Enhancements
- Implement data augmentation to improve model robustness.
- Use pre-trained models like ResNet or VGG for transfer learning.
- Optimize hyperparameters for better accuracy.

---
This project provides a solid foundation for image classification tasks using deep learning in PyTorch. 
Modify and experiment to enhance performance further!

