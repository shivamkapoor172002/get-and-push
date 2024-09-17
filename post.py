import cv2
import mediapipe as mp
import pyautogui
import time  # For adding delay

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

                # Extract key points of hand landmarks (index finger tip and thumb tip)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate Euclidean distance between thumb and index finger
                thumb_index_distance = ((thumb_tip.x - index_finger_tip.x) ** 2 +
                                        (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5

                # If the thumb and index finger are close enough (gesture 'O')
                if thumb_index_distance < 0.05:
                    # Hide or minimize the OpenCV window
                    cv2.destroyWindow("Hand Gesture Recognition")
                    
                    # Add a slight delay to ensure the window is fully minimized
                    time.sleep(0.5)

                    # Take a screenshot using PyAutoGUI
                    print("O gesture detected. Taking screenshot...")
                    screenshot = pyautogui.screenshot()
                    screenshot.save(r"C:\Users\shiva\OneDrive\Desktop\screenshot\images\screenshot_o_gesture.png")
                    
                    # Optionally, wait and show the window again after taking the screenshot
                    time.sleep(1)
                    cv2.namedWindow("Hand Gesture Recognition", cv2.WINDOW_NORMAL)

        # Display the webcam feed with hand landmarks
        cv2.imshow("Hand Gesture Recognition", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
