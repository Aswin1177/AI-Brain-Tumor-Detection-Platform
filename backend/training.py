from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
import pandas as pd
import cv2
import os

datadir="/Users/aswinsanthosh/Desktop/project2/classification_task/train"
categories=[]
for i in os.listdir(datadir):
    categories.append(i)

X=[]
y=[]
for category in categories:
    print("loading",category,"...")
    path=os.path.join(datadir,category)
    for img in os.listdir(path):
        image=cv2.imread(os.path.join(path,img))
        image=cv2.resize(image,(224,224))
        X.append(image)
        y.append(categories.index(category))
X_train=np.array(X)
y_train=np.array(y)
X_train=preprocess_input(X_train)

base_model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable=False

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(units=1024,activation='relu')(x)
x=Dense(units=512,activation='relu')(x)
x=Dense(units=256,activation='relu')(x)
x=Dense(units=128,activation='relu')(x)
output=Dense(units=4,activation='softmax')(x)

model=Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=32,epochs=3)
model.save("Brain_tumour_model.h5")