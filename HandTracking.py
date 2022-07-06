import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mphand = mp.solutions.hands
hand = mphand.Hands()
mpdraw = mp.solutions.drawing_utils

prev_time = 0
curr_time = 0

while True:
    success, img = cap.read()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hand.process(rgb_img)
    # print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handmk in result.multi_hand_landmarks:
            for id, l in enumerate(handmk.landmark):
                #print(i,l)
                height, width, channels = img.shape
                cx, cy = int(l.x * width), int(l.y * height)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx,cy), 15, (255, 0, 255), cv2.FILLED)

            mpdraw.draw_landmarks(img, handmk, mphand.HAND_CONNECTIONS)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img,str(int(fps)), (10,70),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
