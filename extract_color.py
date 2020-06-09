import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)

while(1):
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV - Hue Saturation Brightness
    #  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #  g2rgb = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_white = np.array([0,0,160], dtype=np.uint8)
    upper_white = np.array([172,111,255], dtype=np.uint8)

    #  lower_gray = np.array([0,0,120], dtype=np.uint8)
    #  lower_gray = np.array([0,0,195], dtype=np.uint8)
    #  upper_white = np.array([0,0,255], dtype=np.uint8)

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_white, upper_white)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask= mask)

    cv.imshow('hsv',hsv)
    #  cv.imshow('g2rgb',g2rgb)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
