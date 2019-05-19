import cv2
import os
import sys

# Get user supplied values

# in opencv/data/haarcascades

casBasePath = 'haarcascades'

cascPaths = ("haarcascade_frontalface_default.xml",
             "haarcascade_frontalcatface_extended.xml",
             "haarcascade_profileface.xml",
             "haarcascade_fullbody.xml",
             "haarcascade_upperbody.xml",
             "haarcascade_frontalface_alt.xml")

def face_detection(cascPath, imagePath = sys.argv[1]):
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)
    
    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        # It is not used for a new cascade.
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    
    if not len(faces):
        return False, image
    msg = "Found {0} faces in {1} [{2}]".format(len(faces),
                                                imagePath,
                                                cascPath)
    print(msg)
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return cv2, image

for i in cascPaths:
    cascPath = os.path.join(casBasePath, i)
    imagePath = sys.argv[1]
    cv2, image = face_detection(cascPath, imagePath)
    if cv2:
        cv2.imshow(imagePath, image)
        cv2.waitKey(0)
        break
