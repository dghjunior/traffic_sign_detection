from PIL import Image

for i in range(877):
    im = Image.open("data/images/road" + str(i) + ".png")
    f = open('data/yolo_annotations/road' + str(i) + '.txt', 'r')
    box = f.readline()
    count = 1
    while box != '':
        box = box.split(' ')
        x = int(box[1])
        y = int(box[2])
        w = int(box[3])
        h = int(box[4])
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        im1 = im.crop((x1, y1, x2, y2))
        im1.save("data/cropped_images/road" + str(i) + '_' + box[0] + '-' + str(count) + ".png")
        box = f.readline()