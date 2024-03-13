import os
import numpy as np
import pickle #allows us to use this model in the future in projects etc

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

#train classifier
classifier = SVC()

#here we will train 12 image classifiers (built from the crossproduct of gamma and c)
#we will then choose the best from all the classifiers we trained
parameters = [{'gamma':[0.01,0.001,0.0001],"C":[1,10,100,1000]}]

gridSearch = GridSearchCV(classifier,parameters)

gridSearch.fit(xTrain,yTrain)

#test performance
bestEstimator = gridSearch.best_estimator_
yPrediction = bestEstimator.predict(xTest)
score = accuracy_score(yPrediction, yTest)
print('{}% of samples were correctly classified'.format(str(score*100)))

pickle.dump(bestEstimator,open('./model.p','wb'))