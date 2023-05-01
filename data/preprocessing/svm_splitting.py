from PIL import Image
from bs4 import BeautifulSoup

for i in range(877):
    im = Image.open("data/images/road" + str(i) + ".png")
    f = open('data/voc_annotations/road' + str(i) + '.xml', 'r')
    bs_data = BeautifulSoup(f.read(), 'lxml')

    objects = bs_data.find_all('object')
    count = 0
    for obj in objects:
        count += 1
        category = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        im_crop = im.crop((xmin, ymin, xmax, ymax))
        im_crop.save('svm/data/cropped_images/' + category + '/road' + str(i) + '_' + str(count) + '.png')