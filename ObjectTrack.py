import cv2
import numpy as np

from TeloMain import height

# Set the width and height of the video frame
frameWidth = 640
frameHeight = 480
# Initialize video capture (use camera index 1)
cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

deadZone = 100

global imgContour

# Empty callback function for trackbar (used as a placeholder)
def empty(a):
    pass

# Create a window for HSV trackbars
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
# Create trackbars for adjusting HSV values
cv2.createTrackbar("HUE Min", "HSV", 19, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 35, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 107, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("Value Min", "HSV", 89, 255, empty)
cv2.createTrackbar("Value Max", "HSV", 255, 255, empty)

# Create a window for parameter trackbars
cv2.namedWindow("parametrs")
cv2.resizeWindow("parametrs", 640, 240)
# Create trackbars for adjusting edge detection parameters
cv2.createTrackbar("Threshold1", "Parameters", 166, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 171, 255, empty)
cv2.createTrackbar("Area", "Parameters", 3759, 30000, empty)

# Function to stack multiple images in a single window
def stackImages(scale, imgArray):
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
    for x in range(0, rows):
        if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)4
        else:
            imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
        if len(imgArray[x].shape) == 2:
            imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
    hor = np.hstack(imgArray)
    ver = hor
    return ver

# Function to get and draw contours around detected objects
def getContours(img, imgContour):
    # Find contours in the given image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Get minimum area from trackbar
        areaMin = cv2.getTrackbarPos("Area", "Parameters")

        if area > areaMin:
            # Draw contours if the area is greater than the minimum
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            print(len(approx))

            # Draw bounding box around contour
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 5)

            # Display the number of points of the contour
            cv2.putText(imgContour, "Points: " + str(len(approx)),
                        (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

            # Display the area of the contour
            cv2.putText(imgContour, "Area: " + str(int(area)),
                        (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

            # Display the coordinates of the contour
            cv2.putText(imgContour, str(int(x)) + " : " + str(int(y)),
                        (x - 20, y - 45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

            # Calculate the center of the bounding box
            cx = int(x + (w / 2))
            cy = int(y + (h / 2))

            # Display directions based on the position of the center
            if cx < int(frameWidth / 2) - deadZone:
                cv2.putText(imgContour, "GO LEFT", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (0, int(frameHeight / 2) - deadZone),
                              (int(frameWidth / 2) - deadZone, int(frameHeight / 2) + deadZone),
                              (0, 0, 255), cv2.FILLED)

            elif cx > int(frameWidth / 2) + deadZone:
                cv2.putText(imgContour, "GO RIGHT", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (int(frameWidth / 2) + deadZone, int(frameHeight / 2) - deadZone),
                              (frameWidth, int(frameHeight / 2) + deadZone),
                              (0, 0, 255), cv2.FILLED)

            elif cy < int(frameHeight / 2) - deadZone:
                cv2.putText(imgContour, "GO UP", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (int(frameWidth / 2) - deadZone, 0),
                              (int(frameWidth / 2) + deadZone, int(frameHeight / 2) - deadZone),
                              (0, 0, 255), cv2.FILLED)

            elif cy > int(frameHeight / 2) + deadZone:
                cv2.putText(imgContour, "GO DOWN", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(imgContour, (int(frameWidth / 2) - deadZone, int(frameHeight / 2) + deadZone),
                              (int(frameWidth / 2) + deadZone, frameHeight),
                              (0, 0, 255), cv2.FILLED)

            # Draw a line from the center of the frame to the center of the contour
            cv2.line(imgContour, (int(frameWidth / 2), int(frameHeight / 2)), (cx, cy), (0, 0, 255), 3)

# Function to display directional lines and indicators on the frame
def display(img):
    # Draw vertical lines to indicate dead zones
    cv2.line(img, (int(frameWidth / 2) - deadZone, 0), (int(frameWidth / 2) - deadZone, frameHeight), (255, 255, 0), 3)
    cv2.line(img, (int(frameWidth / 2) + deadZone, 0), (int(frameWidth / 2) + deadZone, frameHeight), (255, 255, 0), 3)

    # Draw a circle in the center of the frame
    cv2.circle(img, (int(frameWidth / 2), int(frameHeight / 2)), 5, (0, 0, 255), 5)

    # Draw horizontal lines to indicate dead zones
    cv2.line(img, (0, int(frameHeight / 2) - deadZone), (frameWidth, int(frameHeight / 2) - deadZone), (255, 255, 0), 3)
    cv2.line(img, (0, int(frameHeight / 2) + deadZone), (frameWidth, int(frameHeight / 2) + deadZone), (255, 255, 0), 3)

# Main loop to read frames from the camera and process them
while True:
    # Read a frame from the video capture
    _, img = cap.read()
    # Create a copy of the frame to draw contours on
    imgContour = img.copy()
    # Convert the frame to HSV color space
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get the current positions of all HSV trackbars
    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    print(h_min)

    # Create lower and upper HSV bounds
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # Create a mask to filter out specific colors
    mask = cv2.inRange(imgHSV, lower, upper)
    # Apply the mask to the original frame
    result = cv2.bitwise_and(img, img, mask=mask)
    # Convert the mask to a BGR image
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Apply Gaussian blur to the result
    imgBlur = cv2.GaussianBlur(result, (7, 7), 1)
    # Convert the blurred image to grayscale
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    # Get threshold values from trackbars
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    # Apply Canny edge detection
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)

    # Define a kernel for dilation
    kernel = np.ones((5, 5))
    # Apply dilation to the edges
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    # Find and draw contours on the frame
    getContours(imgDil, imgContour)
    # Display directional indicators on the frame
    display(imgContour)

    # Stack images for display
    stack = stackImages(0.7, ([img, result], [imgDil, imgContour]))

    # Show the stacked images
    cv2.imshow('Horizontal Stacking', stack)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release the video capture and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
