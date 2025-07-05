import cv2 as cv
import numpy as np
import time
import os
import HandTrackingModule_Final as htm
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt

############################################
blue = (255, 0, 255)
green = (0, 255, 0)
red = (0, 0, 255)
Eraser = (0,0,0)
BoxWidthGlobal = 120
CurrentColor = blue
xp, yp = 0, 0
combined = np.zeros((480, 960, 3), np.uint8)
image_path = 'hand_image_2.jpg'
############################################
def DrawHeaderImage(width=480, height=480, BoxWidth = BoxWidthGlobal, Selector=blue):
    """
    This function draws the header to be displayed on top of webcam feed.
    """
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
    FinalImg = cv.putText(img, "RESET", (BoxWidth * 3 + 30, 40), cv.FONT_HERSHEY_PLAIN, 2.0, red)
    return FinalImg

def preprocess_image(image_path):
    """
    Preprocess a hand-drawn digit image to match MNIST format
    """
    # Load image
    img = Image.open(image_path)

    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')

    # Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img)

    # Invert colors if needed (MNIST has white digits on black background)
    # Check if background is mostly white (hand-drawn on white paper)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalize to 0-1 range
    img_array = img_array.astype('float32') / 255.0

    # Reshape for model input
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


# Alternative: Load pre-trained model if you already have one saved
def load_and_predict(model_path, image_path):
    """
    Load a pre-trained model and predict digit
    """
    try:
        loaded_model = keras.models.load_model(model_path)
        processed_img = preprocess_image(image_path)
        prediction = loaded_model.predict(processed_img)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        return predicted_digit, confidence
    except Exception as e:
        print(f"Error loading model or predicting: {e}")
        return None, None

cap = cv.VideoCapture(0)
detector = htm.handDetector()
canvas = np.zeros((480, 480, 3), np.uint8)

while True:
    ret, img = cap.read()
    img = cv.flip(img, 1)
    img = cv.resize(img, [480, 480])
    img[0:50, 0:480] = DrawHeaderImage(480, 50, BoxWidthGlobal, CurrentColor)



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
                    img[0:50, 0:480]=DrawHeaderImage(480, 50, BoxWidthGlobal, 3)
                    # CurrentColor = Eraser
                    canvas = np.zeros((480, 480, 3), np.uint8)
                    combined = np.zeros((480, 960, 3), np.uint8)



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
    # print(img.shape)
    # print(imgInv.shape)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img, canvas)
    # cv.imshow("canvas", canvas)
    # cv.imshow("frame", img)

    combined[0:480, 0: 480] = img
    combined[0: 480, 480: 960] = canvas
    cv.imshow("Display", combined)

    key = cv.waitKey(1)
    if key == ord('s'):  # Press 's' to save
        cv.imwrite("hand_image.jpg", canvas)
        print("Image saved!")
    elif key == ord('a'):  # Press 's' to apply cnn algo
        temp_path = 'temp.jpg'
        cv.imwrite("temp.jpg", canvas)
        predicted_digit, confidence = load_and_predict('digit_recognition_model.h5', 'temp.jpg')
        print(f"Predicted digit: {predicted_digit}")
        print(f"Confidence: {confidence:.2f}")
        canvas = cv.putText(canvas,f"{predicted_digit} ({confidence:.2f}) ", (350, 460), cv.FONT_HERSHEY_PLAIN, 1.0, red)




