# CIFAR-10 Image Classifier

This project implements a deep learning model to classify images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN). It demonstrates various techniques like data augmentation, model optimization, and hyperparameter tuning to improve classification performance.

## Project Overview
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. This project uses a custom CNN model to classify these images with an emphasis on improving accuracy.

## Features
- Data Augmentation: To prevent overfitting and improve generalization.
- CNN Architecture: A deep learning model with multiple convolutional layers.
- Model Training: Implemented with Adam optimizer and Cross-Entropy loss.
- Evaluation: Performance is evaluated using accuracy and loss curves.

## Setup
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/CIFAR-10-Image-Classifier.git
    cd CIFAR-10-Image-Classifier
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the CIFAR-10 dataset by running the `main.py` script.

## Usage
Run the `main.py` script to train and evaluate the model:
```bash
python main.py

