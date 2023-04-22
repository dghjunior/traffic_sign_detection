import os
import shutil
from sklearn.model_selection import train_test_split

def split_images():
    data = os.listdir('archive/images')
    train, test = train_test_split(data, test_size=0.2, random_state=27)

    for file in train:
        shutil.copy('archive/images/' + file, 'archive/train/' + file)

    for file in test:
        shutil.copy('archive/images/' + file, 'archive/test/' + file)

def split_labels():
    train = os.listdir('archive/train')
    test = os.listdir('archive/test')

    for file in train:
        shutil.copy('archive/labels/' + file[:-3] + 'txt', 'archive/train/' + file[:-3] + 'txt')

    for file in test:
        shutil.copy('archive/labels/' + file[:-3] + 'txt', 'archive/test/' + file[:-3] + 'txt')

# split_images()

split_labels()