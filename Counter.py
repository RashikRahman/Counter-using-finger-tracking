import cv2
import time
import os
from glob import glob
import HandTrackModule as htm

wCam, hCam = 640,480

cap = cv2.VideoCapture(0) #,cv2.CAP_DSHOW
cap.set(3,wCam)
cap.set(4,hCam)

OverlayFingerImages = [cv2.resize(cv2.imread(file),(130,180)) for file in glob("./Fingers/*.jpg")]
detector = htm.handDetector(detectionCon=0.8)

pTime = 0
tipIds = [4, 8, 12, 16, 20]

while True:
    totalFingers=0
    success, img = cap.read()
    totalFingers = detector.fingerUp(img)
    print(totalFingers)
    img[0:180,0:130] = OverlayFingerImages[totalFingers-1]
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, f'FPS: {int(fps)}',(400,70),
        cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow('Image',img)
    cv2.waitKey(1)

