import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from cnn_model import CNNModel
from utils.data_augmentation import get_transforms
from utils.model_training import train_model
from utils.evaluation import evaluate_model

# Load dataset
transform = get_transforms()
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Initialize model, criterion, and optimizer
model = CNNModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Evaluate the model
evaluate_model(model, test_loader)

