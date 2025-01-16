# Vision Transformer for Chest X-Ray Analysis  

![ViT_Transformer_Architecture (1)](https://github.com/user-attachments/assets/e6b62242-4d39-45a2-b5e8-14910378fbd1)


## Overview  
This repository contains the implementation of a Vision Transformer (ViT) model for the classification and analysis of chest X-ray images. The model segments images into patches, incorporates positional encoding, and uses multi-head self-attention layers to capture long-range dependencies. It is tailored for multi-label classification tasks, making it suitable for detecting multiple thoracic diseases from X-rays.  

## Key Features  
- **Patch Embedding**: Input X-ray images are divided into non-overlapping patches and projected into an embedding space.  
- **Transformer Encoder**: A stack of transformer layers with multi-head self-attention and feedforward networks for robust feature learning.  
- **Multi-Label Classification**: Outputs probabilities for multiple disease classes using a fully connected layer with sigmoid activation.  

## Model Architecture  
The proposed architecture processes chest X-rays through the following stages:  
1. **Patch Extraction and Embedding**  
2. **Positional Encoding**  
3. **Transformer Encoder Layers**  
4. **Classification Head**  

Refer to the model diagram above for a detailed visual representation.  

## Dataset  
The model is trained and evaluated on chest X-ray datasets such as **NIH Chest X-ray 14** or similar publicly available datasets.  

## Requirements  
- Python 3.8+  
- PyTorch 1.9+  
- NumPy, Matplotlib, and other dependencies listed in `requirements.txt`.  

## Usage  

### Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/Sourodip-ghosh123/ViT-vs-DenseNet121.git  
   cd vit-chest-xray  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

### Training  
1. Prepare the dataset and adjust the configuration in `config.yaml`.  
2. Train the model:  
   ```bash  
   python train.py  
   ```  

### Inference  
To run inference on new X-ray images:  
```bash  
python inference.py --image-path /path/to/image  
```  

## Results  
Our results indicate that Vision Transformers (ViT) outperform DenseNet in accuracy and AUC for most pathologies. For instance, the AUC for Cardiomegaly is 0.9150 for ViT compared to 0.9126 for DenseNet, and for Emphysema, the AUC is 0.9380 for ViT, compared to 0.9360 for DenseNet.

## Acknowledgments  
Special thanks to the open-source community and contributors of pre-trained ViT models.  

## License  
This project is licensed under the MIT License. See the `LICENSE` file for details.  
