import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
# in opencv/data/haarcascades
#cascPath = "haarcascade_frontalface_default.xml"
cascPath = "haarcascades/haarcascade_frontalcatface_extended.xml"
#cascPath = "haarcascade_frontalface_alt.xml"


# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

if not len(faces): sys.exit(1)
msg = "Found {0} faces in {1}".format(len(faces), imagePath)
print(msg)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow(msg, image)
cv2.waitKey(0)
