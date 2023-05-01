import os
import shutil
from sklearn.model_selection import train_test_split

def split_images():
    data = os.listdir('data/images')
    train, test = train_test_split(data, test_size=0.2, random_state=27)

    for file in train:
        shutil.copy('data/images/' + file, 'data/train/' + file)

    for file in test:
        shutil.copy('data/images/' + file, 'data/test/' + file)

def split_labels():
    train = os.listdir('data/train')
    test = os.listdir('data/test')

    for file in train:
        shutil.copy('data/labels/' + file[:-3] + 'txt', 'data/train/' + file[:-3] + 'txt')

    for file in test:
        shutil.copy('data/labels/' + file[:-3] + 'txt', 'data/test/' + file[:-3] + 'txt')

# split_images()

# split_labels()