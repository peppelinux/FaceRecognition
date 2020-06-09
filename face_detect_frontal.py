import cv2
import os
import random
import numpy as np
import sys

# Get user supplied values
# in opencv/data/haarcascades

casBasePath = 'haarcascades'

cascPaths = (
             "haarcascade_frontalface_default.xml",
             #  "haarcascade_frontalcatface_extended.xml",
             #  "haarcascade_profileface.xml",
             #  "haarcascade_fullbody.xml",
             #  "haarcascade_upperbody.xml",
             #  "haarcascade_frontalface_alt.xml",
             #  "haarcascade_eye_tree_eyeglasses.xml",
             #  "haarcascade_eye.xml"
             )

def filter_dimensions(image, faces):
    # check that the image is not too much greater than the detected face
    image_rap = image.shape[:2]
    face = faces[0]
    # it's a rectangle!
    face_rap = face[3], face[3]

    rap_x = '{:01.3f}'.format( face_rap[0] / image_rap[0])
    rap_y = '{:01.3f}'.format( face_rap[1] / image_rap[1])
    min_rapp = 0.5
    if float(rap_x) < min_rapp or float(rap_x) < min_rapp:
        print("The Face is too small ({}) in relation to the size of "
              "the image {}. This must be at least {}".format(face_rap,
                                                              image_rap,
                                                              min_rapp))
        sys.exit(1)


def filter_n_faces(faces, n=1):
    n_faces = len(faces)
    if not n_faces:
        print("No face detected")
        return False, image
    if n_faces > n:
        print("Found more than 1 faces, invalid.")
        sys.exit(1)


def face_detection(casc, imagePath = sys.argv[1]):
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(casc)

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.01,
        minNeighbors=5,
        # the size of the returning rectangle
        #  minSize=(360, 360),
        minSize=(30, 30),
    )


    msg = "Found {0} faces in {1} [{2}]".format(len(faces),
                                                imagePath,
                                                casc)
    print(msg)

    filter_n_faces(faces)
    filter_dimensions(image, faces)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        img = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return cv2, image


def thresh_callback(val=100):
    threshold = val
    # Detect edges using Canny
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    # Show in a window
    cv2.imshow('Contours', drawing)


for i in cascPaths:
    print('Processing: {}'.format(i))
    cascPath = os.path.join(casBasePath, i)
    imagePath = sys.argv[1]
    res, image = face_detection(cascPath, imagePath)
    if res:
        res.imshow(imagePath, image)
        res.waitKey(0)
        cv2.destroyAllWindows()
        #  break
