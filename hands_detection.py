import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

while True:
    # Read camera feed
    success, frame = cap.read()
    # Flip the frame
    frame_flipped = cv2.flip(frame,1)
    # Convert BGR frame to RGB (Required for the mediapipe model)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break