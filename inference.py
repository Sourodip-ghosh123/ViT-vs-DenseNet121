#!/usr/bin/env python
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Vision Transformer model (consistent with train.py)
class ViTModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTModel, self).__init__()
        self.model = models.vit_b_16(pretrained=False)  # set pretrained to False; weights will be loaded
        if hasattr(self.model.heads, 'head'):
            in_features = self.model.heads.head.in_features
        else:
            in_features = self.model.heads.in_features
        self.model.heads = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def main():
    parser = argparse.ArgumentParser(description="Inference using Vision Transformer Model")
    parser.add_argument('--image-path', type=str, required=True,
                        help="Path to the input image")
    parser.add_argument('--model-path', type=str, required=True,
                        help="Path to the trained model file")
    parser.add_argument('--num-classes', type=int, default=4,
                        help="Number of classes")
    args = parser.parse_args()

    # Define image transforms (should match those used in training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image = Image.open(args.image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Initialize the model and load trained weights
    model = ViTModel(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model-path, map_location=device))
    model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        predicted_class = probs.argmax(dim=1).item()
    
    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {probs.cpu().numpy()}")

if __name__ == '__main__':
    main()
