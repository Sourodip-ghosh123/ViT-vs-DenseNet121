#!/usr/bin/env python
import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Vision Transformer model (using torchvision’s ViT implementation)
class ViTModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTModel, self).__init__()
        self.model = models.vit_b_16(pretrained=True)
        # Depending on your torchvision version, the head structure may differ:
        if hasattr(self.model.heads, 'head'):
            in_features = self.model.heads.head.in_features
        else:
            in_features = self.model.heads.in_features
        self.model.heads = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Training function (calls the previously defined train_model)
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
    training_time = time.time() - start_time
    print(f"Training complete in {training_time:.0f} seconds")
    return model

# Evaluation function to compute ROC-AUC for each class
def evaluate_model(model, dataloader, num_classes):
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            # Use softmax to convert logits into probabilities
            probs = F.softmax(outputs, dim=1)
            all_outputs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Convert integer labels to one-hot encoding
    all_labels_onehot = np.eye(num_classes)[all_labels]
    auc_scores = []
    for i in range(num_classes):
        try:
            auc = roc_auc_score(all_labels_onehot[:, i], all_outputs[:, i])
        except ValueError:
            auc = float('nan')
        auc_scores.append(auc)
    for i, auc in enumerate(auc_scores):
        print(f"Class {i} ROC-AUC: {auc:.4f}")
    mean_auc = np.nanmean(auc_scores)
    print(f"Mean ROC-AUC: {mean_auc:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train Vision Transformer Model")
    parser.add_argument('--data-dir', type=str, required=True,
                        help="Path to training data directory (structured for ImageFolder)")
    parser.add_argument('--val-dir', type=str, default=None,
                        help="Path to validation data directory (optional)")
    parser.add_argument('--output-model', type=str, default="vit_model.pth",
                        help="Path to save the trained model")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size for training")
    parser.add_argument('--num-classes', type=int, default=4,
                        help="Number of classes")
    args = parser.parse_args()

    # Define transforms for training (adjust as needed)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Prepare the training dataset and loader
    train_dataset = datasets.ImageFolder(root=args.data_dir, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Optional: prepare validation loader for ROC-AUC evaluation
    if args.val_dir:
        val_dataset = datasets.ImageFolder(root=args.val_dir, transform=train_transforms)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_loader = None

    # Initialize model, loss, and optimizer
    model = ViTModel(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Starting training...")
    model = train_model(model, train_loader, criterion, optimizer, args.epochs)

    if val_loader:
        print("Evaluating model on validation set...")
        evaluate_model(model, val_loader, args.num_classes)

    # Save the trained model
    torch.save(model.state_dict(), args.output_model)
    print(f"Model saved to {args.output_model}")

if __name__ == '__main__':
    main()
