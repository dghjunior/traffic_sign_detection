import cv2
import time
import os
import imutils
import argparse
import numpy as np
from PIL import Image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--prototxt', default='caffe/SSD_MobileNet_prototxt.txt')
ap.add_argument('-m', '--model', default='caffe/SSD_MobileNet.caffemodel')
ap.add_argument('-c', '--confidence', type=float, default=0.7)
args = vars(ap.parse_args())

labels = ["trafficlight", "speedlimit", "crosswalk", "stop"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

print('[Status] Loading Model ...')
nn = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

for image in os.listdir('caffe/data/train/'):
    image = cv2.imread('caffe/data/train/' + image)
    resized = imutils.resize(image, width=400)
    (h, w) = resized.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(resized, (300, 300)), 0.007843, (300, 300), 127.5)

    nn.setInput(blob)
    detections = nn.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args['confidence']:
            idx = int(detections[0, 0, i, 1])

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            label = '{}: {:.2f}%'.format(labels[idx], confidence * 100)
            cv2.rectangle(resized, (startX, startY), (endX, endY), colors[idx], 2)

            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(resized, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    cv2.imshow('Output', resized)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
cv2.destroyAllWindows()