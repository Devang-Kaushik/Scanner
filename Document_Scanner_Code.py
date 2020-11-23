from cv2 import cv2
import numpy as np
import mapper


input = cv2.imread("Input_File.jpg")
input = cv2.resize(input, (1300, 800))
copy = input.copy()

#converting to Gray-Scale
grayscale = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)


#applying GaussianBlur
gaussian = cv2.GaussianBlur(grayscale, (5, 5), 0)

#applying Canny-Edge-Detection Technique
cannyedge=cv2.Canny(gaussian, 30, 50)

contours, hierarchy = cv2.findContours(cannyedge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
for i in contours:
    x = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02*x, True)

    if len(approx) == 4:
        a = approx
        break
approx=mapper.mapp(a)

window = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])

top_view =cv2.getPerspectiveTransform(approx, window)
wrap = cv2.warpPerspective(copy, top_view, (800, 800))


cv2.imshow("Scanned Image", wrap)

