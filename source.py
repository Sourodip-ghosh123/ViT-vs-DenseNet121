import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os import listdir
from os.path import join, isfile, isdir
from glob import glob

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

from PIL import Image
sns.set()
from tqdm import tqdm
%matplotlib inline

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

all_xray_df = pd.read_csv("../input/chestxray8-dataframe/train_df.csv")
all_xray_df.drop(['No Finding'], axis = 1, inplace = True)
all_xray_df.head()

data_dir1 = '../input/data/'
data_dir2 = '../input/chestxray8-dataframe/'
train_df = pd.read_csv(data_dir1 + 'Data_Entry_2017.csv')
image_label_map = pd.read_csv(data_dir2 + 'train_df.csv')
bad_labels = pd.read_csv(data_dir2 + 'cxr14_bad_labels.csv')

# Listing all the .jpg filepaths
image_paths = glob(data_dir1+'images_*/images/*.png')
print(f'Total image files found : {len(image_paths)}')
print(f'Total number of image labels: {image_label_map.shape[0]}')
print(f'Unique patients: {len(train_df["Patient ID"].unique())}')

image_label_map.drop(['No Finding'], axis = 1, inplace = True)
labels = image_label_map.columns[2:-1]
labels

all_xray_df = all_xray_df[89485:]

labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']
          
train_df.rename(columns={"Image Index": "Index"}, inplace = True)
image_label_map.rename(columns={"Image Index": "Index"}, inplace = True)
train_df = train_df[~train_df.Index.isin(bad_labels.Index)]
train_df.shape

Index =[]
for path in image_paths:
    Index.append(path.split('/')[5])
index_path_map = pd.DataFrame({'Index':Index, 'FilePath': image_paths})
index_path_map.head()

pd.merge(train_df, index_path_map, on='Index', how='left')

pd.merge(train_df, index_path_map, on='Index', how='left')

IMAGE_SIZE=[256, 256]
EPOCHS = 20
BATCH_SIZE = 64

def get_generator(df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w = 320, target_h = 320):
    
    print("getting testing generators...")
    
    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)
    
    test_generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    
    return test_generator

train_generator = get_generator(df = image_label_map,
                                      image_dir = None, 
                                      x_col = 'FilePath',
                                      y_cols = labels, 
                                      batch_size=BATCH_SIZE,
                                      target_w = IMAGE_SIZE[0], 
                                      target_h = IMAGE_SIZE[1] 
                                      )
                                      
IMAGE_DIR = ""
test_generator= get_generator(all_xray_df, "", "FilePath", labels)

X, Y = train_generator.next()

def get_label(y):

    ret_labels = []
    for idx in range(len(y)):
        if y[idx]: ret_labels.append(labels[idx])
    if len(ret_labels):  return '|'.join(ret_labels)
    else: return 'No Label'

rows = int(np.floor(np.sqrt(X.shape[0])))
cols = int(X.shape[0]//rows)
fig = plt.figure(figsize=(20,15))
for i in range(1, rows*cols+1):
    fig.add_subplot(rows, cols, i)
    plt.imshow(X[i-1], cmap='gray')
    plt.title(get_label(Y[i-1]))
    plt.axis(False)
    fig.add_subplot

from statistics import mean
auc_rocs, thresholds, sensitivity, specificity, accuracy, precision, recall, f1 = get_roc_curve(labels, predicted_vals, test_generator)
from tabulate import tabulate
table = zip(labels, auc_rocs)
print(f"Mean AUC : {mean(auc_rocs)}")
print(tabulate(table, headers = ['Pathology', 'AUC'], tablefmt = 'fancy_grid'))
from tabulate import tabulate
table = zip(labels, auc_rocs, thresholds, sensitivity, specificity, accuracy, precision, recall, f1)
print(tabulate(table, headers = ['Pathology', 'AUC', 'Threshold Value', 'Sensitivity', 'Specificity', 'Accuracy', 'Precision', 'Recall', 'F1 Score'], tablefmt = 'fancy_grid'))
print(predicted_vals[0:100])

#Densenet
base_model = DenseNet121(weights='../input/chestxray8-dataframe/densenet.hdf5', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
predicted_vals = model.predict(test_generator, steps = len(test_generator), verbose = 1)

#Vision Transformer
class ViTModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTModel, self).__init__()
        self.model = models.vit_b_16(pretrained=True)
        self.model.heads = nn.Linear(self.model.heads.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
vit_model = ViTModel(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_vit = optim.Adam(vit_model.parameters(), lr=1e-4)

def train_model(model, train_loader, optimizer, num_epochs=5):
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    training_time = time.time() - start_time
