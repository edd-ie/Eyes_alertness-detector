import numpy as np
import cv2

# Initializing face & eye cascade classifiers from xml file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Face
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')  # Eye

# Variable store execution state
first_read = True

# Start video capture
cap = cv2.VideoCapture(0)
ret, img = cap.read()

while (ret):
    """ Start algorithm """
    ret, img = cap.read()
    # Convert recording to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Bilateral filtering - remove impurities
    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    # Detect the face for ROI
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # ROI_face is input to eye detection
            roi_face = gray[y:y + h, x:x + w]
            roi_face_clr = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(50, 50))

            # Determining if eyes are open
            if len(eyes) >= 2:
                # Check if detection is running
                if (first_read):
                    cv2.putText(img, "Eye dectected, Press s to start",
                                (70, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 255, 0), 2)
                else:
                    cv2.putText(img, "Eyes open!",
                                (70, 70), cv2.FONT_HERSHEY_PLAIN, 2,
                                (255, 255, 255), 2)

            else:
                if first_read:
                    # Ensuring eyes are present
                    cv2.putText(img, "No eyes detected",
                                (70, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 2)
                else:
                    # Blinking detected, rerun algorithm
                    print("Blink detected--------------")
                    cv2.waitKey(3000)
                    first_read = True

    else:
        cv2.putText(img,
                    "No face detected", (100, 100),
                    cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 0), 2)

    # Controling algorithm with keys
    cv2.imshow('img', img)
    a = cv2.waitKey(1)
    if a == ord('q'):
        break
    elif a == ord('s') and first_read:
        # This will start the detection
        first_read = False

cap.release()
cv2.destroyAllWindows()
