"""
Hand Tracking & Digit Recognition System:

This algorithm is a real-time digit recognition system that allows users to draw digits in the air using hand gestures, with CNN-based digit classifications.

Requirements: OpenCV, NumPy, TensorFlow, PIL (Pillow), HandTrackingModule_Final (custom module)

Controls:
- Index + Middle finger up: Selection mode (choose color/reset)
- Index finger only: Drawing mode
- Press 's': Save current canvas
- Press 'a': Predict drawn digit using CNN
- Press 'ESC': Quit application
"""
import os
import cv2 as cv
import numpy as np
from PIL import Image
from tensorflow import keras
import HandTrackingModule_Final as htm

# ====================================
# Configuration and Global Variables
# ====================================
BOX_WIDTH = 120
COLORS = {
    "blue": (255, 0, 255),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "eraser": (0, 0, 0)
}
current_color = COLORS["blue"]
xp, yp = 0, 0
canvas = np.zeros((480, 480, 3), np.uint8)
combined_display = np.zeros((480, 960, 3), np.uint8)

# ====================================
# Helper Functions
# ====================================

def draw_header_image(width = 480, height = 50, selector = (255, 0, 255)):
    """Draws the color selection header."""
    img = np.zeros((height, width, 3), np.uint8)
    for i, color in enumerate(COLORS.values()):
        cv.rectangle(img, (i * BOX_WIDTH, 0), ((i+1)*BOX_WIDTH, height), color, -1)

    selected_index = list(COLORS.values()).index(selector)
    cv.circle(img, (selected_index * BOX_WIDTH + BOX_WIDTH // 2, 25), 10, (0, 0, 0), cv.FILLED)
    cv.putText(img, "RESET", (BOX_WIDTH * 3 + 30, 40), cv.FONT_HERSHEY_PLAIN, 2.0, COLORS["red"])
    return img

def preprocess_image(image_path):
    """Preprocesses an image for MNIST model input."""
    img = Image.open(image_path).convert('L').resize((28, 28), Image.LANCZOS)
    img_array = np.array(img)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    img_array = img_array.astype('float32') / 255.0
    return img_array.reshape(1, 28, 28, 1)

def load_and_predict(model_path, image_path):
    """Loads a model and returns digit prediction and confidence."""
    try:
        model = keras.models.load_model(model_path)
        processed = preprocess_image(image_path)
        prediction = model.predict(processed)
        return np.argmax(prediction), float(np.max(prediction))
    except Exception as e:
        print(f"[ERROR] Model prediction failed: {e}")
        return None, None

# ================================
# Main Loop
# ================================
def main():
    global xp, yp, current_color, canvas, combined_display

    cap = cv.VideoCapture(0)
    detector = htm.handDetector()

    while True:
        ret, img = cap.read()
        if not ret:
            print("[ERROR] Camera not found.")
            break

        img = cv.flip(img, 1)
        img = cv.resize(img, (480, 480))
        img[0:50, 0:480] = draw_header_image(selector=current_color)

        # Detect hand landmarks
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if lmList[0]:
            lmEl = lmList[0]
            x1, y1 = lmEl[8][1:]
            x2, y2 = lmEl[12][1:]
            fingers = detector.fingersUp()

            # Color selection mode
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                if y1 < 70:
                    idx = x1 // BOX_WIDTH
                    if idx == 0:
                        current_color = COLORS["blue"]
                    elif idx == 1:
                        current_color = COLORS["green"]
                    elif idx == 2:
                        current_color = COLORS["red"]
                    elif idx == 3:
                        canvas = np.zeros((480, 480, 3), np.uint8)
                        combined_display = np.zeros((480, 960, 3), np.uint8)

            # Drawing mode
            elif fingers[1] and not fingers[2]:
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                thickness = 80 if current_color == COLORS["eraser"] else 20
                radius = 40 if current_color == COLORS["eraser"] else 20
                cv.circle(img, (x1, y1), radius, current_color, cv.FILLED)
                cv.line(canvas, (xp, yp), (x1, y1), current_color, thickness)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0

        # Combine layers
        imgGray = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
        _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
        imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
        img = cv.bitwise_and(img, imgInv)
        img = cv.bitwise_or(img, canvas)

        combined_display[:, :480] = img
        combined_display[:, 480:] = canvas
        cv.imshow("Digit Drawer", combined_display)

        key = cv.waitKey(1)
        if key == ord('s'):
            cv.imwrite("hand_image.jpg", canvas)
            print("[INFO] Image saved as hand_image.jpg")

        elif key == ord('a'):
            temp_path = "temp.jpg"
            cv.imwrite(temp_path, canvas)
            digit, confidence = load_and_predict("digit_recognition_model.h5", temp_path)
            if digit is not None:
                print(f"[RESULT] Predicted Digit: {digit} | Confidence: {confidence:.2f}")
                canvas = cv.putText(canvas, f"{digit} ({confidence:.2f})", (350, 460),
                                    cv.FONT_HERSHEY_PLAIN, 1.0, COLORS["red"])

        elif key == 27:  # ESC to exit
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()