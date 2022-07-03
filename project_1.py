import cv2
import numpy as np
from matplotlib import pyplot as plt
#https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_russian_plate_number.xml
nPlateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
capture = cv2.VideoCapture(0)
frameW = 640
frameH = 480 
capture.set(3, frameW)
capture.set(4, frameH)
capture.set(10,150)
count = 0

minArea = 1200
color = (0, 0, 255)

while True:
	success, img = capture.read()
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)

	for (x, y, w, h) in numberPlates:
		area = w*h
		if area > minArea:
			
			cv2.rectangle(img, (x, y), (x + w + 10, y + h + 10), (0, 0, 255), 1)
			cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2 )
			
			imgRoi = img[y:y-5+h+7, x:x-5+w+7]

			cv2.imshow("ROI", imgRoi)

	cv2.imshow("Live Result", img)
	if cv2.waitKey(1) & 0xFF == ord('s'):
		cv2.imwrite("NoPlate"+str(count)+".jpg", imgRoi)
		cv2.rectangle(img, (0,200), (640,300),(0,255,0),cv2.FILLED)
		cv2.putText(img,"Image Saved", (150,265),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255),2)
		cv2.imshow("Result",img)
		cv2.waitKey(500)
		count += 1
	elif cv2.waitKey(1) & 0xFF == ord('q'):
		break 