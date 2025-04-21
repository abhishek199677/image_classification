# image_classification using CNN



#### A TensorFlow/Keras CNN model for multi-class image classification, featuring a 6-layer convolutional architecture with max-pooling, dropout regularization, and automatic data loading from directories

# 📌 Overview
This repository contains a Convolutional Neural Network (CNN) implemented in TensorFlow/Keras for image classification tasks. The model takes 128x128 RGB images as input and classifies them into multiple categories using a deep learning architecture.

# 🛠️ Technical Implementation

## Model Architecture
6 Convolutional Blocks with increasing filter depth (32 → 64 → 128 → 254)

Each Conv layer uses:

3×3 kernels with ReLU activation

'same' padding to preserve dimensions

Followed by 2×2 MaxPooling for dimensionality reduction

Flattened output fed into:

128-unit Dense layer with ReLU

30% Dropout for regularization

Final softmax layer for multi-class classification

## Data Pipeline
Uses ImageDataGenerator for efficient loading

Automatic class detection from directory structure

Image normalization (pixel values scaled to 0-1)

Supports batch processing (batch size = 16)

## Training Configuration
Optimizer: Adam

Loss: Categorical Crossentropy

Metrics: Accuracy

10 Epochs with validation monitoring


## 📂 Directory Structure
data/
  train/       # Training images (organized by class)
  val/         # Validation images (organized by class)
image_cnn_classifier.h5  # Saved model


## 🚀 Usage
Organize images in data/train and data/val with class-specific subdirectories

Run the script to train and save the model

Model will be saved as image_cnn_classifier.h5


