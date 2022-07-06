import cv2
import mediapipe as mp
import time


class handTracker():
    def __init__(self, mode=False, maxhands=2, modelC =1, detcon=0.5, trcon=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.modelC = modelC
        self.detcon = detcon
        self.trcon = trcon
        self.mphand = mp.solutions.hands
        self.hand = self.mphand.Hands(self.mode, self.maxhands, self.modelC, self.detcon, self.trcon)
        self.mpdraw = mp.solutions.drawing_utils

    def find(self, img, draw=True):

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hand.process(rgb_img)
        # print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for handmk in self.result.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handmk, self.mphand.HAND_CONNECTIONS)
        return img



    def findPos(self, img, handnum = 0, draw = True):

        lmk = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handnum]


            for id, l in enumerate(myHand.landmark):
                # print(i,l)
                height, width, channels = img.shape
                cx, cy = int(l.x * width), int(l.y * height)
                #print(id, cx, cy)
                lmk.append([id,cx,cy])
                if draw:
                # if id == 0:
                   cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

        return lmk



def main():
    prev_time = 0
    curr_time = 0
    cap = cv2.VideoCapture(0)
    detector = handTracker()

    while True:
        success, img = cap.read()
        img = detector.find(img)
        lmk = detector.findPos(img)
        if len(lmk) != 0:
            print(lmk[4])
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
