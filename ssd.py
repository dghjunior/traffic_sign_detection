import cv2
import time
import imutils
import argparse
import numpy as np

from imutils.video import FPS
from imutils.video import VideoStream

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.7, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

labels = ["trafficlight", "speedlimit", "crosswalk", "stop"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

print('[Status] Loading model...')
nn = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

print('[Status] Starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    nn.setInput(blob)
    detections = nn.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)

            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    fps.update()

fps.stop()

print("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[Info] Approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()