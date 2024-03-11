import os

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import numpy as np

#type imageClassifier\Scripts\activate to run the virtual environment
#and also python -m venc imageClassifier if it needs creating again


#prepare
inputDir = "C:/Users/izzym/OneDrive/Documents/GitHub/imageClassifier"
categories = ["empty","notEmpty"]

data = []
labels = []
for categoryIdx,category in enumerate(categories):
    for file in os.listdir(os.path.join(inputDir,category)):
        imgPath = os.path.join(inputDir, category,file)
        img = imread(imgPath)
        img = resize(img,(15,15))
        data.append(img.flatten())
        labels.append(categoryIdx)


data = np.asarray(data)
labels = np.asarray(labels)


#train/test split
xTrain, xTest, yTrain, yTest = train_test_split(data,labels,test_size=0.2,shuffle=True,stratify=labels)
#continue from 15:23


#train classifier


#test performance