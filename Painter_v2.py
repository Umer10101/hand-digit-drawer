import cv2 as cv
import numpy as np
import time
import os
import HandTrackingModule_Final as htm
from tensorflow.keras.models import load_model

############################################
blue = (255, 0, 255)
green = (0, 255, 0)
red = (0, 0, 255)
Eraser = (0,0,0)
BoxWidthGlobal = 160
CurrentColor = blue
xp, yp = 0, 0
combined = np.zeros((480, 1280, 3), np.uint8)
############################################
# model = load_model("https://huggingface.co/keras-io/mnist-convnet/resolve/main/mnist_model.h5")
def DrawHeaderImage(width=640, height=480, BoxWidth = BoxWidthGlobal, Selector=blue):


    blank = np.zeros((height,width, 3), np.uint8)
    img = cv.rectangle(blank, (0, 0), (BoxWidth, 50), blue, -1)
    img = cv.rectangle(img, (BoxWidth, 0), (BoxWidth*2, 50), green, -1)
    img = cv.rectangle(img, (BoxWidth*2, 0), (BoxWidth*3, 50), red, -1)
    img = cv.rectangle(img, (BoxWidth*3, 0), (BoxWidth*4, 50), Eraser, -1)

    if Selector == blue:
        img = cv.circle(img, (BoxWidth//2, 25), 10, (0,0,0), cv.FILLED)
    elif Selector == green:
        img = cv.circle(img, (BoxWidth+ BoxWidth//2, 25), 10, (0,0,0), cv.FILLED)
    elif Selector == red:
        img = cv.circle(img, (BoxWidth*2+ BoxWidth//2, 25), 10, (0,0,0), cv.FILLED)
    elif Selector == Eraser:
        img = cv.circle(img, (BoxWidth*3+ BoxWidth//2, 25), 10, (0,0,0), cv.FILLED)
    FinalImg = cv.putText(img, "ERASE", (BoxWidth * 3 + 30, 40), cv.FONT_HERSHEY_PLAIN, 2.0, red)
    return FinalImg



cap = cv.VideoCapture(0)
detector = htm.handDetector()
canvas = np.zeros((480, 640, 3), np.uint8)

while True:
    ret, img = cap.read()
    img = cv.flip(img, 1)
    img[0:50, 0:640] = DrawHeaderImage(640, 50, BoxWidthGlobal, CurrentColor)



    # Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    # Get tip of index and middle finger:
    if len(lmList[0]) != 0:
        lmEl=lmList[0]
        x1, y1 = lmEl[8][1:]
        x2, y2 = lmEl[12][1:]


        finger = detector.fingersUp()


        if finger[1] and finger[2]:
            xp, yp = 0, 0
            if y1 < 70:

                if x1< BoxWidthGlobal:
                    CurrentColor = blue

                elif x1 < 2 * BoxWidthGlobal:

                    CurrentColor = green
                elif x1 < 3 * BoxWidthGlobal:
                    CurrentColor = red

                else:
                    img[0:50, 0:640]=DrawHeaderImage(640, 50, BoxWidthGlobal, 3)
                    CurrentColor = Eraser



        elif finger[1] and (not finger[2]):

            if xp ==0 and yp==0:
                xp = x1; yp = y1
            if CurrentColor == Eraser:
                cv.circle(img, (x1, y1), 40, CurrentColor, cv.FILLED)
                cv.line(canvas, (xp, yp), (x1,y1), CurrentColor, 80)
            else:
                cv.circle(img, (x1, y1), 20, CurrentColor, cv.FILLED)
                cv.line(canvas, (xp, yp), (x1, y1), CurrentColor, 20)
            xp, yp = x1, y1

        else:
            xp, yp = 0, 0


    imgGray = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)

    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, canvas)
    # cv.imshow("canvas", canvas)
    # cv.imshow("frame", img)

    combined[0:480, 0: 640] = img
    combined[0: 480, 640: 1280] = canvas
    cv.imshow("Display", combined)

    key = cv.waitKey(1)
    if key == ord('s'):  # Press 's' to save
        cv.imwrite("hand_image.jpg", canvas)
        print("Image saved!")




