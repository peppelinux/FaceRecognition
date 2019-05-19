# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2
import sys
import time


#cap = cv2.VideoCapture('http://192.168.3.35:8081')
cap = cv2.VideoCapture(sys.argv[1])

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

print('Running ...')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    if not len(faces):
        time.sleep(0.2)
        continue

    print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
