import cv2
import mediapipe as mp
import pyautogui
import os
from datetime import datetime

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Folder where the screenshot images are saved
image_folder = r"C:\Users\shiva\OneDrive\Desktop\screenshot\images"

# Function to get the latest image from the folder
def get_latest_image(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png")]
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    return latest_file

# Function to check if the "V" gesture is shown (index and middle fingers extended)
def is_v_gesture(hand_landmarks):
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

    # Condition: index and middle finger are higher (y-coordinate smaller) than the rest
    if (index_finger_tip.y < ring_finger_tip.y and
        middle_finger_tip.y < ring_finger_tip.y and
        ring_finger_tip.y > thumb_tip.y and
        pinky_tip.y > thumb_tip.y):
        return True
    return False

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Create an instance for hand detection
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB as MediaPipe uses RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand landmarks
        result = hands.process(rgb_frame)

        # Draw landmarks if hands are detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if the "V" gesture is detected
                if is_v_gesture(hand_landmarks):
                    print("V gesture detected. Displaying the latest screenshot...")

                    # Get the latest image from the folder
                    latest_image = get_latest_image(image_folder)

                    if latest_image:
                        # Load and display the image
                        screenshot_image = cv2.imread(latest_image)
                        cv2.imshow("Latest Screenshot", screenshot_image)
                    else:
                        print("No screenshot found in the folder!")

        # Display the webcam feed with hand landmarks
        cv2.imshow("Hand Gesture Recognition", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
