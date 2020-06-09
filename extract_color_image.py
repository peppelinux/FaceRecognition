import cv2 as cv
import numpy as np
import sys

imagePath = sys.argv[1]
frame = cv.imread(cv.samples.findFile(imagePath))
#cap = cv.VideoCapture(0)

# Convert BGR to HSV - Hue Saturation Brightness
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
g2rgb = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
hsv = cv.cvtColor(g2rgb, cv.COLOR_BGR2HSV)

# define range of blue color in HSV
#  lower_white = np.array([0,0,160], dtype=np.uint8)
#  upper_white = np.array([172,111,255], dtype=np.uint8)

lower_gray = np.array([0,0,200], dtype=np.uint8)
upper_white = np.array([0,0,255], dtype=np.uint8)

# Threshold the HSV image to get only blue colors
mask = cv.inRange(hsv, lower_gray, upper_white)

# mask = cv.inRange(hsv, lower_white, upper_white)

# Bitwise-AND mask and original image
res = cv.bitwise_and(frame, frame, mask= mask)

cv.imshow('hsv',hsv)
#  cv.imshow('g2rgb',g2rgb)
#  cv.imshow('mask',mask)
cv.imshow('res',res)

cv.waitKey(0)
cv.destroyAllWindows()
