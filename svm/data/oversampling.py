from PIL import Image
import os

categories=["trafficlight", "crosswalk", "stop"]
data = os.listdir('svm/data/train/')

def oversample():
    for category in categories:
        print(f'loading... category : {category}')
        path=os.path.join('svm/data/train/',category)
        for img in os.listdir(path):
            f = Image.open(os.path.join(path,img))
            # Flip image horizontally
            f = f.transpose(Image.FLIP_LEFT_RIGHT)

            # Save flipped image
            f.save(os.path.join(path, 'flipped_'+img))

def duplicate():
    for category in categories:
        print(f'loading... category : {category}')
        path=os.path.join('svm/data/train/',category)
        for img in os.listdir(path):
            f = Image.open(os.path.join(path,img))
            # Flip image horizontally

            # Save flipped image
            f.save(os.path.join(path, 'copy_'+img))

duplicate()