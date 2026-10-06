import cv2
import mediapipe as mp


# MediaPipe classes
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand detector settings
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Create detector
detector = HandLandmarker.create_from_options(options)

# Open webcam
camera = cv2.VideoCapture(0)

timestamp = 0

while True:

    success, frame = camera.read()

    if not success:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp += 1

    # Detect hand
    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    # If hand detected
    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        print("------ HAND ------")

        points = []

        # Get all 21 landmarks
        for i, landmark in enumerate(hand):

            # Print coordinates
            print(
                i,
                "X =", landmark.x,
                "Y =", landmark.y,
                "Z =", landmark.z
            )

            # Convert normalized coordinates to pixels
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            points.append((x, y))

        # Connections between landmarks
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        # Draw lines
        for start, end in connections:

            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )

        # Draw dots
        for point in points:

            cv2.circle(
                frame,
                point,
                5,
                (0, 0, 255),
                -1
            )

        cv2.putText(
            frame,
            "HAND DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO HAND",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Show webcam
    cv2.imshow(
        "Sign Language Recognition",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
detector.close()
cv2.destroyAllWindows()