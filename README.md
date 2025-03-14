# Vision Transformer for Chest X-Ray Analysis

![ViT_Transformer_Architecture (1)](https://github.com/user-attachments/assets/e6b62242-4d39-45a2-b5e8-14910378fbd1)

## Overview
This repository contains the implementation of a Vision Transformer (ViT) model for the classification and analysis of chest X-ray images. The model segments images into patches, incorporates positional encoding, and uses multi-head self-attention layers to capture long-range dependencies. It is designed for multi-label classification tasks, making it suitable for detecting multiple thoracic diseases from X-rays.

This updated version of the repository includes complete training and inference pipelines. It addresses previous issues with missing scripts and includes an optional ROC-AUC evaluation step during training.

## Dataset Overview
This NIH Chest X-ray Dataset is comprised of 112,120 X-ray images with disease labels from 30,805 unique patients. To create these labels, the authors used Natural Language Processing to text-mine disease classifications from the associated radiological reports. The labels are expected to be >90% accurate and suitable for weakly-supervised learning. The original radiology reports are not publicly available but you can find more details on the labeling process in this Open Access paper: "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." (Wang et al.)

Link to paper: https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community


### Data limitations:
The image labels are NLP extracted so there could be some erroneous labels but the NLP labeling accuracy is estimated to be >90%.
Very limited numbers of disease region bounding boxes (See BBox_list_2017.csv)
Chest x-ray radiology reports are not anticipated to be publicly shared. Parties who use this public dataset are encouraged to share their “updated” image labels and/or new bounding boxes in their own studied later, maybe through manual annotation

File contents
Image format: 112,120 total images with size 1024 x 1024

images_001.zip: Contains 4999 images

images_002.zip: Contains 10,000 images

images_003.zip: Contains 10,000 images

images_004.zip: Contains 10,000 images

images_005.zip: Contains 10,000 images

images_006.zip: Contains 10,000 images

images_007.zip: Contains 10,000 images

images_008.zip: Contains 10,000 images

images_009.zip: Contains 10,000 images

images_010.zip: Contains 10,000 images

images_011.zip: Contains 10,000 images

images_012.zip: Contains 7,121 images

README_ChestXray.pdf: Original README file

BBox_list_2017.csv: Bounding box coordinates. Note: Start at x,y, extend horizontally w pixels, and vertically h pixels

Image Index: File name
Finding Label: Disease type (Class label)
Bbox x
Bbox y
Bbox w
Bbox h
Data_entry_2017.csv: Class labels and patient data for the entire dataset

Image Index: File name
Finding Labels: Disease type (Class label)
Follow-up #
Patient ID
Patient Age
Patient Gender
View Position: X-ray orientation
OriginalImageWidth
OriginalImageHeight
OriginalImagePixelSpacing_x
OriginalImagePixelSpacing_y

### Class descriptions
There are 15 classes (14 diseases, and one for "No findings"). Images can be classified as "No findings" or one or more disease classes:

Atelectasis
Consolidation
Infiltration
Pneumothorax
Edema
Emphysema
Fibrosis
Effusion
Pneumonia
Pleural_thickening
Cardiomegaly
Nodule Mass
Hernia

### Full Dataset Content
There are 12 zip files in total and range from ~2 gb to 4 gb in size. Additionally, we randomly sampled 5% of these images and created a smaller dataset for use in Kernels. The random sample contains 5606 X-ray images and class labels.

Sample: sample.zip

### Modifications to original data
Original TAR archives were converted to ZIP archives to be compatible with the Kaggle platform

CSV headers slightly modified to be more explicit in comma separation and also to allow fields to be self-explanatory

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
