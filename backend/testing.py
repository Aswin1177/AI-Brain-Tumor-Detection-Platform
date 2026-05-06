import pandas as pd
import numpy as np
import os
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
datadir="/Users/aswinsanthosh/Desktop/project2/classification_task/test"
categories=['pituitary', 'no_tumor', 'glioma', 'meningioma']
model=load_model("/Users/aswinsanthosh/Desktop/project2/classification_task/Brain_tumour_model.h5")
X_img=[]
y_cat=[]
for category in categories:
    print("Loading",category,"...")
    path=os.path.join(datadir,category)
    for img in os.listdir(path):
        resimg=cv2.imread(os.path.join(path,img))
        resimg=cv2.resize(resimg,(224,224))
        X_img.append(resimg)
        y_cat.append(categories.index(category))
X_test=np.array(X_img)
y_test=np.array(y_cat)
X_test=preprocess_input(X_test)
pred=model.predict(X_test)
y_pred=np.argmax(pred,axis=1)

print("Accuracy Score : ", accuracy_score(y_test,y_pred))
print("Classification Report : ", classification_report(y_test,y_pred))