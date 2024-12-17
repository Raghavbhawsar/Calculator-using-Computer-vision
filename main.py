import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

import cv2
import mediapipe as mp
import numpy as np

center_point = None
end_point = None
shape = None

def calculate_circle_area(radius):
    return np.pi * (radius ** 2)

def calculate_square_area(side_length):
    return side_length ** 2

def detect_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    index_tip = (landmarks[8].x, landmarks[8].y)
    thumb_tip = (landmarks[4].x, landmarks[4].y)
    middle_tip = (landmarks[12].x, landmarks[12].y)

    # Check if the index finger is up and the thumb is not close to it
    if landmarks[8].y < landmarks[6].y and np.linalg.norm(
            np.array([landmarks[8].x, landmarks[8].y]) - np.array([landmarks[4].x, landmarks[4].y])) > 0.1:
        return "DRAW_CIRCLE", index_tip

    # Check if the middle finger is up and the thumb is not close to it
    if landmarks[12].y < landmarks[10].y and np.linalg.norm(
            np.array([landmarks[12].x, landmarks[12].y]) - np.array([landmarks[4].x, landmarks[4].y])) > 0.1:
        return "DRAW_SQUARE", middle_tip

    # Check if the thumb is close to the index finger tip to complete the drawing
    if np.linalg.norm(np.array([landmarks[8].x, landmarks[8].y]) - np.array([landmarks[4].x, landmarks[4].y])) < 0.05:
        return "COMPLETE", index_tip

    return "MOVE", None

def main():
    global center_point, end_point, shape

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    canvas = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Flip the image horizontally for a natural webcam view
        frame = cv2.flip(frame, 1)

        # Initialize the canvas with the same size as the frame
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Convert the image color back so it can be displayed properly
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture, point = detect_gesture(hand_landmarks)

                if gesture in ["DRAW_CIRCLE", "DRAW_SQUARE"]:
                    shape = gesture
                    if center_point is None:
                        center_point = (int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0]))
                    else:
                        end_point = (int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0]))
                        if shape == "DRAW_CIRCLE":
                            radius = int(np.linalg.norm(np.array(center_point) - np.array(end_point)))
                            canvas = np.zeros_like(frame)  # Clear the canvas for each new shape
                            cv2.circle(canvas, center_point, radius, (255, 255, 255), 2)
                            area = calculate_circle_area(radius)
                            cv2.putText(canvas, f"Circle Area: {area:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        elif shape == "DRAW_SQUARE":
                            side_length = int(np.linalg.norm(np.array(center_point) - np.array(end_point)) * np.sqrt(2))
                            top_left = (center_point[0] - side_length // 2, center_point[1] - side_length // 2)
                            bottom_right = (center_point[0] + side_length // 2, center_point[1] + side_length // 2)
                            canvas = np.zeros_like(frame)  # Clear the canvas for each new shape
                            cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), 2)
                            area = calculate_square_area(side_length)
                            cv2.putText(canvas, f"Square Area: {area:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                elif gesture == "COMPLETE":
                    center_point = None
                    end_point = None

        combined = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)
        cv2.imshow('Virtual Math Solver', combined)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
