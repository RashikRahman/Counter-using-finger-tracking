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
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)

        img[0:180,0:130] = OverlayFingerImages[totalFingers-1]
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, f'FPS: {int(fps)}',(400,70),
        cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    cv2.imshow('Image',img)
    cv2.waitKey(1)

