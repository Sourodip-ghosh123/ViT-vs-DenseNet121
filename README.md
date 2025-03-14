# Vision Transformer for Chest X-Ray Analysis

![ViT_Transformer_Architecture (1)](https://github.com/user-attachments/assets/e6b62242-4d39-45a2-b5e8-14910378fbd1)

## Overview
This repository contains the implementation of a Vision Transformer (ViT) model for the classification and analysis of chest X-ray images. The model segments images into patches, incorporates positional encoding, and uses multi-head self-attention layers to capture long-range dependencies. It is designed for multi-label classification tasks, making it suitable for detecting multiple thoracic diseases from X-rays.

This updated version of the repository includes complete training and inference pipelines. It addresses previous issues with missing scripts and includes an optional ROC-AUC evaluation step during training.

## Key Features
- **Patch Embedding:** Divides input X-ray images into non-overlapping patches and projects them into an embedding space.
- **Transformer Encoder:** Employs a stack of transformer layers with multi-head self-attention and feedforward networks for robust feature learning.
- **Multi-Label Classification:** Uses a fully connected layer with sigmoid activation to output probabilities for multiple disease classes.
- **Training & Inference:** Provides complete `train.py` and `inference.py` scripts.
- **ROC-AUC Evaluation:** Optionally evaluates the model using ROC-AUC metrics on a validation set during training.

## Model Architecture
The proposed architecture processes chest X-rays through the following stages:
1. **Patch Extraction and Embedding**
2. **Positional Encoding**
3. **Transformer Encoder Layers**
4. **Classification Head**

Refer to the diagram above for a visual representation of the model.

## Dataset
The model is trained and evaluated on chest X-ray datasets such as **NIH Chest X-ray 14** or other similar publicly available datasets.

## Requirements
- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib
- Other dependencies as listed in `requirements.txt`

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sourodip-ghosh123/ViT-vs-DenseNet121.git
   cd ViT-vs-DenseNet121

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage
### Training
The repository now includes a complete training pipeline. To train the Vision Transformer model:

### Prepare Your Dataset
Organize your training data in a format compatible with PyTorch’s ImageFolder (i.e., one folder per class). Optionally, prepare a separate validation dataset for ROC-AUC evaluation.

### Run the Training Script
```
python train.py --data-dir /path/to/train_data --val-dir /path/to/val_data --epochs 10 --batch-size 32 --num-classes 4
```

### Inference
To perform inference on new X-ray images:
```
python inference.py --image-path /path/to/image.jpg --model-path vit_model.pth --num-classes 4
```
The inference script loads the trained model, preprocesses the input image, and outputs the predicted class along with the associated probabilities.

## Results  
Our results indicate that Vision Transformers (ViT) outperform DenseNet in accuracy and AUC for most pathologies. For instance, the AUC for Cardiomegaly is 0.9150 for ViT compared to 0.9126 for DenseNet, and for Emphysema, the AUC is 0.9380 for ViT, compared to 0.9360 for DenseNet.

## Acknowledgments  
Special thanks to the open-source community and contributors of pre-trained ViT models.  

## License  
This project is licensed under the MIT License. See the `LICENSE` file for details.  
