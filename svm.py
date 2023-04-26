import numpy as np
import pandas as pd
import os
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.preprocessing import MinMaxScaler

import PIL
import cv2
import pickle

sign_paths = glob('archive/images/*')
neg_paths = []

for class_path in glob("archive/natural_images"+"/*"):
    if class_path != "archive/natural_images/car":
        paths = random.choices(glob(class_path+"/*"), k=125)
        neg_paths = paths + neg_paths

print("There are {} sign images in the dataset".format(len(sign_paths)))
print("There are {} negative images in the dataset".format(len(neg_paths)))

example_image = rgb2gray(np.asarray(PIL.Image.open(sign_paths[0]))[:,:,:3])
hog_features, visualized = hog(example_image,orientations=9,pixels_per_cell=(16,16),
                              cells_per_block=(2,2),
                              visualize=True
                             )
fig = plt.figure(figsize=(12,6))
fig.add_subplot(1,2,1)
plt.imshow(example_image)
plt.axis("off")
fig.add_subplot(1,2,2)
plt.imshow(visualized, cmap="gray")
plt.axis("off")
plt.show()

hog_features.shape

pos_images = []
neg_images = []

pos_labels = np.ones(len(sign_paths))
neg_labels = np.zeros(len(neg_paths))

start = time.time()

for sign_path in sign_paths:
    img = np.asarray(PIL.Image.open(sign_path))
    img = cv2.cvtColor(cv2.resize(img, (96, 64)), cv2.COLOR_RGB2GRAY)
    img = hog(img, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2))

    pos_images.append(img)

for neg_path in neg_paths:
    img = np.asarray(PIL.Image.open(neg_path))
    img = cv2.cvtColor(cv2.resize(img, (96, 64)), cv2.COLOR_RGB2GRAY)
    img = hog(img, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2))

    neg_images.append(img)

x = np.asarray(pos_images + neg_images)
y = np.asarray(list(pos_labels) + list(neg_labels))

processTime = round(time.time() - start, 2)
print("Reading images and extracting features has taken {} sedons".format(processTime))

print("Shape of image set", x.shape)
print("Shape of labels", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=27)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)
print("Accuracy score of model is ", accuracy_score(y_pred=y_pred, y_true=y_test)*100)

def slideExtract(image, windowSize=(96, 64), channel="RGB", step=12):
    if channel == "RGB":
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif channel == "BGR":
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif channel.lower()!="grayscale" or channel.lower()!="gray":
        raise Exception("Invalid channel type")
    
    coords = []
    features = []

    hIm, wIm = image.shape[:2]

    for w1, w2 in zip(range(0, wIm-windowSize[0], step), range(windowSize[0], wIm, step)):
        for h1, h2 in zip(range(0, hIm-windowSize[1], step), range(windowSize[1], hIm, step)):
            window = img[h1:h2, w1:w2]
            features_of_window = hog(window, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2))
            coords.append((w1, w2, h1, h2))
            features.append(features_of_window)
    return (coords, np.asarray(features))

example_image = np.asarray(PIL.Image.open('archive/images/road10.png'))
coords, features = slideExtract(example_image, channel="RGB")

coords[:5]

features.shape

class Heatmap():
    def __init__(self, original_image):
        self.mask = np.zeros(original_image.shape[:2])

    def incValOfReg(self, coords):
        w1, w2, h1, h2 = coords
        self.mask[h1:h2, w1:w2] = self.mask[h1:h2, w1:w2] + 30

    def decValOfReg(self, coords):
        w1, w2, h1, h2 = coords
        self.mask[h1:h2, w1:w2] = self.mask[h1:h2, w1:w2] - 30

    def compileHeatmap(self):
        scaler = MinMaxScaler()
        self.mask = scaler.fit_transform(self.mask)
        self.mask = np.asarray(self.mask*255).astype(np.uint8)
        self.mask = cv2.inRange(self.mask, 170, 255)
        return self.mask
    
def detect(image):
    coords, features = slideExtract(image)
    htmp = Heatmap(image)

    for i in range(len(features)):
        decision = svc.predict([features[i]])
        if decision[0] == 1:
            htmp.incValOfReg(coords[i])
        else:
            htmp.decValOfReg(coords[i])
    
    mask = htmp.compileHeatmap()

    cont, _ = cv2.findContours(mask, 1, 2)[:2]
    for c in cont:
        if cv2.contourArea(c) < 70*70:
            continue
        
        (x, y, w, h) = cv2.boundingRect(c)
        image = cv2.rectangle(image, (x, y), (x+w, y+h), (255), 2)

    return image

detected = detect(np.asarray(PIL.Image.open('archive/images/road1.png')))
plt.imshow(detected)
plt.show()