import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('panda.jpg', 0) 
ret, binary_img = cv2.threshold(img, 127, 255, 0)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

erosi = cv2.erode(binary_img, kernel, iterations=1)

dilasi = cv2.dilate(binary_img, kernel, iterations=1)

opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(img, cmap='gray')
plt.title('Gambar Asli (Grayscale)')

plt.subplot(232)
plt.imshow(binary_img, cmap='gray')
plt.title('Citra Biner (Threshold)')

plt.subplot(233)
plt.imshow(erosi, cmap='gray')
plt.title('Erosi')

plt.subplot(234)
plt.imshow(dilasi, cmap='gray')
plt.title('Dilasi')

plt.subplot(235)
plt.imshow(opening, cmap='gray')
plt.title('Opening')

plt.subplot(236)
plt.imshow(closing, cmap='gray')
plt.title('Closing')

plt.tight_layout()
plt.show()