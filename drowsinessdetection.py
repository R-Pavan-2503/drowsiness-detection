import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the minimum and maximum threshold of EAR
ear_min_threshold = 0.25
ear_max_threshold = 0.35

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Grab the frame
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_fBGR2GRAY)

    # Detect faces
    rects = detector(gray, 0)

    # Loop over the face detections
    for rect in rects:
        shape = face_utils.shape_to_np(predictor(gray, rect))

        # Extract the left and right eye coordinates
        leftEye = shape[36:42]
        rightEye = shape[42:48]

        # Compute the average Eye Aspect Ratio for both eyes
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        # Compute the convex hull for both eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # Draw the convex hull for both eyes
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if the average ear is below the minimum threshold
        if ear < ear_min_threshold:
            print("Drowsy")
        elif ear > ear_max_threshold:
            print("Awake")

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' is pressed
    if key == ord("q"):
        break

# Release the video capture
cap