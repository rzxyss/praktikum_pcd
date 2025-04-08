import cv2

img = cv2.imread('panda.jpg')
cv2.imshow('image', img)
cv2.waitKey()
cv2.destoyAllWindows()