import cv2
import cv2 as cv
import mediapipe as mp
import time

class HandGestureModule:
    def __init__(self, mode = False, maxHands=2, comp = 1, minDetn= 0.5, minTrack= 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.comp = comp
        self.minDetn = minDetn
        self.minTrack = minTrack

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(mode, maxHands, comp, minDetn, minTrack)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            if draw:
                for handLms in results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img


def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = HandGestureModule()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv.imshow("Cam", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()