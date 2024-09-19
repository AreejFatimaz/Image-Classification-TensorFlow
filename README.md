# Image-Classification-TensorFlow

## Project Overview
This project focuses on image classification using TensorFlow. It includes loading and exploring the CIFAR-10 dataset, building a convolutional neural network (CNN) for image classification, training the model, and evaluating its performance. The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 classes, making it a standard benchmark for evaluating image classification models.

## Loading the CIFAR-10 Dataset
The project starts by loading the CIFAR-10 dataset, which includes both a training set and a test set. The dataset comprises 32x32 color images across 10 different classes.

## Dataset Overview

### Dataset Classes
The CIFAR-10 dataset is divided into the following 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

### Dataset Split
- **Training Set**: 50,000 images (5,000 per class) used for model training.
- **Test Set**: 10,000 images (1,000 per class) used for model evaluation.

### Dataset Features
- **Image Size**: 32x32 pixels
- **Color Channels**: RGB (Red, Green, Blue)
- **Labels**: Each image is labeled with one of the 10 class categories.

## Exploring and Analyzing the Dataset
Initial exploration includes visualizing sample images, plotting class distributions, and normalizing pixel values to better understand the dataset before model development.

## Creating the Machine Learning Model

### Building the Model
A convolutional neural network (CNN) is designed for image classification. The model architecture incorporates convolutional layers, batch normalization, max-pooling layers, and fully connected layers.

### Compiling the Model
The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function. Accuracy is tracked during training to monitor performance.

## Training the Model

### Initial Training Phase
The model is initially trained on the original CIFAR-10 training dataset for 10 epochs, with performance metrics such as accuracy and loss recorded.

### Training After Data Augmentation
In the second phase, data augmentation techniques like random shifts and horizontal flips are applied to the training data. The model is retrained for an additional 10 epochs, and performance is re-evaluated.

## Evaluating the Model
The model's performance is assessed by tracking accuracy trends and generating a classification report detailing precision, recall, and F1-score for each class.

