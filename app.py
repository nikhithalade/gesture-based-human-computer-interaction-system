import cv2
import mediapipe as mp
import pyautogui
import streamlit as st
import numpy as np

# Streamlit UI Setup
st.title("Gesture-Based Mouse Control")
st.markdown(
    "Move your cursor using hand gestures! Index finger moves the cursor, and a thumb-index pinch performs a click.")

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_x, index_y = 0, 0
prev_x, prev_y = 0, 0
smooth_factor = 0.2  # Smoothing factor to reduce jitter

# Streamlit Video Display
frame_placeholder = st.empty()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.error("Failed to access webcam.")
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index Finger Tip
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y

                    # Apply smoothing
                    index_x = prev_x * (1 - smooth_factor) + index_x * smooth_factor
                    index_y = prev_y * (1 - smooth_factor) + index_y * smooth_factor
                    prev_x, prev_y = index_x, index_y

                    pyautogui.moveTo(index_x, index_y, duration=0.1)

                if id == 4:  # Thumb Tip
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y

                    if abs(index_y - thumb_y) < 20:  # Click gesture
                        pyautogui.click()
                        pyautogui.sleep(1)

    # Display frame in Streamlit
    frame_placeholder.image(frame, channels="BGR", use_column_width=True)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()