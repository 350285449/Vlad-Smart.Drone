import cv2
import numpy as np

from TeloMain import height

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

deadZone = 100

global imgContour

def empty(a):
    pass
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE Min","HSV", 19, 179, empty)
cv2.createTrackbar("HUE Max","HSV", 35, 179, empty)
cv2.createTrackbar("SAT Min","HSV", 107, 255, empty)
cv2.createTrackbar("SAT Max","HSV", 255, 255, empty)
cv2.createTrackbar("Value Min","HSV", 89, 255, empty)
cv2.createTrackbar("Value Max","HSV", 255, 255, empty)

cv2.namedWindow("parametrs")
cv2.resizeWindow("parametrs", 640, 240)
cv2.createTrackbar("Threshold1","Parameters",166, 255,empty)
cv2.createTrackbar("Threshold2","Parameters",171, 255,empty)
cv2.createTrackbar("Area","Parameters",3759, 30000,empty)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvalible = isinstance(imgArray[0], list)
    width = imgArray[0],[0].shape[1]
    height = imgArray[0],[0].shape[0]
    if rowsAvalible:
        for x in range(0, rows):
          for y in range(0, cols):
              if imgArray[x][y].shape[:2] == imgArray[0][0].shape [2]:
                  imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
              else:
                   imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
              if len(imgArray[x][y].shape) == 2:imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
         hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
