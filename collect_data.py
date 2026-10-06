import cv2
import mediapipe as mp
import csv
import os
import time

# --------------------------------
# MediaPipe classes
# --------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# --------------------------------
# Get sign name
# --------------------------------

sign_name = input("Enter sign name: ").strip().upper()

if sign_name == "":
    print("Please enter a sign name.")
    exit()

print()
print("Sign:", sign_name)
print("Press S to start/stop collecting.")
print("Press Q to quit.")


# --------------------------------
# CSV file
# --------------------------------

file_name = "sign_data.csv"

file_exists = os.path.exists(file_name)

file = open(
    file_name,
    "a",
    newline=""
)

writer = csv.writer(file)

# Create header if file is new
if not file_exists:

    header = ["label"]

    for i in range(21):
        header.append("x" + str(i))
        header.append("y" + str(i))
        header.append("z" + str(i))

    writer.writerow(header)


# --------------------------------
# MediaPipe options
# --------------------------------

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


# --------------------------------
# Create detector
# --------------------------------

detector = HandLandmarker.create_from_options(options)


# --------------------------------
# Open webcam
# --------------------------------

camera = cv2.VideoCapture(0)

timestamp = 0

collecting = False

sample_count = 0

last_saved_time = 0

# Save one sample every 0.1 seconds
save_interval = 0.1


# --------------------------------
# Main loop
# --------------------------------

while True:

    success, frame = camera.read()

    if not success:
        print("Camera could not be opened.")
        break

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe image
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


    # --------------------------------
    # If hand detected
    # --------------------------------

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        points = []


        # --------------------------------
        # Get landmark positions
        # --------------------------------

        for landmark in hand:

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            points.append((x, y))


        # --------------------------------
        # Hand connections
        # --------------------------------

        connections = [

            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),

            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),

            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),

            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),

            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),

            (5, 9),
            (9, 13),
            (13, 17)
        ]


        # --------------------------------
        # Draw connections
        # --------------------------------

        for start, end in connections:

            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                2
            )


        # --------------------------------
        # Draw landmark dots
        # --------------------------------

        for point in points:

            cv2.circle(
                frame,
                point,
                5,
                (0, 0, 255),
                -1
            )


        # --------------------------------
        # Save sample
        # --------------------------------

        current_time = time.time()

        if collecting:

            if current_time - last_saved_time >= save_interval:

                row = [sign_name]

                for landmark in hand:

                    row.append(landmark.x)
                    row.append(landmark.y)
                    row.append(landmark.z)

                writer.writerow(row)

                file.flush()

                sample_count += 1

                last_saved_time = current_time


        # --------------------------------
        # Hand detected text
        # --------------------------------

        cv2.putText(
            frame,
            "HAND DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO HAND",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # --------------------------------
    # Display information
    # --------------------------------

    cv2.putText(
        frame,
        "SIGN: " + sign_name,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "S = Start / Stop",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Samples: " + str(sample_count),
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------
    # Show webcam
    # --------------------------------

    cv2.imshow(
        "Sign Language Dataset Collection",
        frame
    )


    # --------------------------------
    # Keyboard controls
    # --------------------------------

    key = cv2.waitKey(1) & 0xFF


    # Start / Stop
    if key == ord("s"):

        collecting = not collecting

        if collecting:

            print("Started collecting:", sign_name)

        else:

            print("Stopped collecting:", sign_name)


    # Quit
    if key == ord("q"):

        break


# --------------------------------
# Close everything
# --------------------------------

file.close()

camera.release()

detector.close()

cv2.destroyAllWindows()

print()
print("Collection finished.")
print("Sign:", sign_name)
print("Samples collected:", sample_count)
print("Saved to:", file_name)