# Indian Traffic Sign Board Classification

This project involves the classification of Indian traffic signs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The dataset includes images of various traffic signs, and the model is trained to recognize and predict their classes.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Setup Instructions](#setup-instructions)
- [Results](#results)
- [Visualizations](#visualizations)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

## Introduction

Traffic sign classification is a critical task for autonomous vehicles and traffic management systems. This project demonstrates the development of a deep learning model to classify traffic signs into their respective categories using labeled images from the **Indian Traffic Sign Dataset**.

## Dataset

The dataset contains:
-**Kaggle Dataset**:(#https://www.kaggle.com/datasets/neelpratiksha/indian-traffic-sign-dataset)
- **Images**: Images of traffic signs organized by class.
- **CSV File**: `traffic_sign.csv`, which includes class labels and their names.

### Preprocessing Steps:
1. Merging duplicate class labels in the CSV file.
2. Combining images from duplicate classes into unified class folders.
3. Resizing images to `32x32` pixels and converting them to RGB format.

## Project Workflow

1. **Dataset Preparation**:
   - Load dataset paths and labels.
   - Handle duplicate class labels and merge them.
   - Preprocess images and prepare data for training.

2. **Model Training**:
   - Split the dataset into training and testing sets (80/20 split).
   - Convert class labels to one-hot encoding.
   - Train a CNN using TensorFlow and Keras.

3. **Evaluation**:
   - Evaluate the model on the test set.
   - Predict classes for test images.

4. **Visualization**:
   - Display test images with their predicted and true class names.

## Model Architecture

The CNN is structured as follows:
- **Convolutional Layers**:
  - Two layers with 32 filters of size `5x5` followed by max pooling.
  - Two layers with 64 filters of size `3x3` followed by max pooling.
- **Dropout Layers**: Added after pooling layers to prevent overfitting.
- **Fully Connected Layers**:
  - One dense layer with 256 neurons and ReLU activation.
  - Output layer with `softmax` activation for classification.

### Optimizer and Loss:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

## Setup Instructions

### Prerequisites
- Python 3.7 or above
- Libraries: TensorFlow, Keras, NumPy, Pandas, OpenCV, Pillow, Matplotlib

### Steps to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/traffic-sign-classification.git
