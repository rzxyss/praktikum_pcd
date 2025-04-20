import cv2
import numpy as np

img = cv2.imread('panda.jpg') 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:

    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])


    sides = len(approx)
    if sides == 3:
        shape_name = "Triangle"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 < aspect_ratio < 1.05:
            shape_name = "Square"
        else:
            shape_name = "Rectangle"
    elif sides == 5:
        shape_name = "Pentagon"
    elif sides == 10:
        shape_name = "Star"
    elif sides >= 8:
        shape_name = "Circle"
    else:
        shape_name = f"{sides}-sided"

    cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)
    cv2.putText(img, shape_name, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2)

cv2.imshow('Identifikasi Bentuk', img)
cv2.waitKey(0)
cv2.destroyAllWindows()