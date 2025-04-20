import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('panda.jpg')  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

threshold_value = 127 
max_value = 255      


ret, thresh_binary = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY)


ret, thresh_binary_inv = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY_INV)


ret, thresh_trunc = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_TRUNC)


ret, thresh_tozero = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_TOZERO)


ret, thresh_tozero_inv = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_TOZERO_INV)


plt.figure(figsize=(15, 10))

plt.subplot(231)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')


plt.subplot(232)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(233)
plt.imshow(thresh_binary, cmap='gray')
plt.title('THRESH_BINARY')


plt.subplot(234)
plt.imshow(thresh_binary_inv, cmap='gray')
plt.title('THRESH_BINARY_INV')


plt.subplot(235)
plt.imshow(thresh_trunc, cmap='gray')
plt.title('THRESH_TRUNC')

plt.subplot(236)
plt.imshow(thresh_tozero, cmap='gray')
plt.title('THRESH_TOZERO')

plt.figure(figsize=(6, 6))
plt.imshow(thresh_tozero_inv, cmap='gray')
plt.title('THRESH_TOZERO_INV')

plt.tight_layout()
plt.show()