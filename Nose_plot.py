import cv2
import dlib
import numpy as np
from imutils.video import FPS
from imutils.video import WebcamVideoStream
import imutils
from imutils import face_utils
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

a = []
b = []
PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()
fa = face_utils.FaceAligner(predictor, desiredFaceWidth=256)

cap=cv2.VideoCapture(0)
im = 0
fps = FPS().start()

while (cap.isOpened()):
    ret,im = cap.read()
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = imutils.resize(im, width=600)

    rects = detector(im, 0)
    if len(rects) != 0:
        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        rect = dlib.rectangle(x, y, x + w, y + h)
        
        get_landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
        for (i, rect) in enumerate(rects):
        	shape = predictor(im, rect)
        	shape = face_utils.shape_to_np(shape)
        	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        		clone = im.copy()
        		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        		for (x, y) in shape[27:35]:
        			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
        		(x, y, w, h) = cv2.boundingRect(np.array([shape[27:35]]))
        		roi = im[y:y + h, x:x + w]
        		print(x,y)
        		roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        		cv2.imshow("NOSE", roi)
        		cv2.imshow("Image", clone)
        		print(roi)
        		a = np.mean(roi) 
        		b.append(a)
        		print(b)
        		print(fps)  
        		win = pg.GraphicsWindow(title="plotting")
        		p1 = win.addPlot(title="FFT")
        		p1.clear()
        		p1.plot(b, pen = 'g')
        		cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
output = face_utils.visualize_facial_landmarks(image, shape)
cv2.imshow("Image", output)
cv2.waitKey(0)