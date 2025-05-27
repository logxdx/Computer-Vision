import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7
) as pose:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break

        # Create black background
        black_bg = np.zeros(frame.shape, dtype=np.uint8)

        # Convert the BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw pose landmarks on the original frame
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Draw pose landmarks on the black background
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                black_bg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # Display both frames
        cv2.imshow("Live Pose Detector", frame)
        cv2.imshow("Pose on Black Background", black_bg)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
