import numpy as np
from sklearn.model_selection import train_test_split
import os

def reorganize():
    # get list of images
    classes = os.listdir("caffe/data/train/")

    # move the train images and labels into the train folder
    with open("caffe/data/train.txt", "w") as f:
        for clazz in classes:
            for image in os.listdir("caffe/data/train/" + clazz):
                os.rename("caffe/data/train/" + clazz + '/' + image, "caffe/data/train/" + classes.index(clazz) + '-' + image)
                f.write(image + ' ' + str(classes.index(clazz)) + '\n')

def split():
    # split the data into train and test sets
    train_images, test_images= train_test_split('caffe/data/train', test_size=0.2, random_state=32)

    with open('caffe/data/train.txt', 'r') as f:
        for image in train_images:
            print()

    # move the test images and labels into the test folder
    with open("caffe/data/test.txt", "w") as f:
        for image in test_images:
            os.rename("caffe/data/train/" + image, "caffe/data/test/" + image)
            f.write(image + '\n')

reorganize()