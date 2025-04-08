import sys
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi
import numpy as np
from matplotlib import pyplot as plt

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('gui_pertemuan4.ui', self)
        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)
        self.actionA.triggered.connect(self.applyConvolutionA)
        self.actionB.triggered.connect(self.applyConvolutionB)
        self.actionI_2.triggered.connect(self.mean_filter_i)
        self.actionII_2.triggered.connect(self.mean_filter_ii)
        self.actionGausian_Filter.triggered.connect(self.gausian_filter)
        self.actionI.triggered.connect(self.shapening_i)
        self.actionII.triggered.connect(self.shapening_ii)
        self.actionIII.triggered.connect(self.shapening_iii)
        self.actionIV.triggered.connect(self.shapening_iv)
        self.actionV.triggered.connect(self.shapening_v)
        self.actionVI.triggered.connect(self.shapening_vi)
        self.actionFilter_Laplace.triggered.connect(self.filter_laplace)
        self.actionMedian_Filter.triggered.connect(self.median_filter)
        self.actionMax_Filter.triggered.connect(self.max_filter)
        self.actionSobel.triggered.connect(self.SobelClicked)
        self.actionCanny.triggered.connect(self.CannyEdgeClicked)

    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('panda.jpg')

    def loadImage(self, flname):
        self.image = cv2.imread(flname)
        self.displayImage(1)

    # D1
    @pyqtSlot()
    def applyConvolutionA(self):
        if self.image is None:
            return

        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        img_filtered = cv2.filter2D(self.image, -1, kernel)
        self.image = img_filtered
        self.displayImage(2)
    
    def applyConvolutionB(self):
        if self.image is None:
            return

        kernel = np.array([[6, 0, -6], [6, 1, -6], [6, 1, -6]])

        img_filtered = cv2.filter2D(self.image, -1, kernel)
        self.image = img_filtered
        self.displayImage(2)

    # D2
    def mean_filter_i(self):
        if self.image is None:
            return
        kernel = (1/9)*np.array([[1,1,1], [1,1,1], [1,1,1]])
        
        if len(self.image.shape) == 3: 
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.image

        hasil = cv2.filter2D(img, -1, kernel)

        plt.imshow(hasil, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def mean_filter_ii(self):
        if self.image is None:
            return
        kernel = (1/4)*np.array([[1,1,1], [1,1,1]])
        
        if len(self.image.shape) == 3: 
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.image

        hasil = cv2.filter2D(img, -1, kernel)
        
        plt.imshow(hasil, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # D3
    def gausian_filter(self):
        if self.image is None:
            return
        kernel_size = 5
        sigma = 1.0

        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        
        if len(self.image.shape) == 3: 
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.image
        
        img_out = cv2.filter2D(img, -1, kernel)
        
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # D4
    def shapening_i(self):
        if self.image is None:
            return
            
        if len(self.image.shape) == 3:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.image.copy()
            
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        
        sharpened = cv2.filter2D(img, -1, kernel)
        
        self.image = sharpened
        self.displayImage(2)

    def shapening_ii(self):
        if self.image is None:
            return
            
        if len(self.image.shape) == 3:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.image.copy()

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        
        sharpened = cv2.filter2D(img, -1, kernel)
        
        self.image = sharpened
        self.displayImage(2)
    
    def shapening_iii(self):
        if self.image is None:
            return
            
        if len(self.image.shape) == 3:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.image.copy()

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        
        sharpened = cv2.filter2D(img, -1, kernel)
        
        self.image = sharpened
        self.displayImage(2)
    
    def shapening_iv(self):
        if self.image is None:
            return
            
        if len(self.image.shape) == 3:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.image.copy()

        kernel = np.array([[1, -2, 1], [-2, 5 -2], [1, -2, 1]])
        
        sharpened = cv2.filter2D(img, -1, kernel)
        
        self.image = sharpened
        self.displayImage(2)
    
    def shapening_v(self):
        if self.image is None:
            return
            
        if len(self.image.shape) == 3:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.image.copy()

        kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, -1]])
        
        sharpened = cv2.filter2D(img, -1, kernel)
        
        self.image = sharpened
        self.displayImage(2)
    
    def shapening_vi(self):
        if self.image is None:
            return
            
        if len(self.image.shape) == 3:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.image.copy()

        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        
        sharpened = cv2.filter2D(img, -1, kernel)
        
        self.image = sharpened
        self.displayImage(2)
    
    def filter_laplace(self):
        if self.image is None:
            return
            
        if len(self.image.shape) == 3:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.image.copy()

        kernel = (1.0 / 16) * np.array([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ])
        
        sharpened = cv2.filter2D(img, -1, kernel)
        
        self.image = sharpened
        self.displayImage(2)
    
    # D5
    def median_filter(self):
        if self.image is None:
            return
            
        if len(self.image.shape) == 3:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.image.copy()
        
        img_out = img.copy()
        h, w = img.shape

        for i in range(3, h-3):
            for j in range(3, w-3):
                neighbors = []

                for k in range(-3, 4):
                    for l in range(-3, 4):
                        a = img[i+k, j+l]
                        neighbors.append(a)
                
                neighbors.sort()
                median = neighbors[24]
                img_out[i, j] = median
        
        self.image = img_out
        self.displayImage(2)
    
    # D6
    def max_filter(self):
        if self.image is None:
            return
            
        if len(self.image.shape) == 3:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.image.copy()

        img_out = img.copy()
        h, w = img.shape

        kernel_size = 3  
        pad = kernel_size // 2  

        for i in range(pad, h-pad):
            for j in range(pad, w-pad):
                neighbors = []

                for k in range(-pad, pad+1):
                    for l in range(-pad, pad+1):
                        a = img[i+k, j+l]
                        neighbors.append(a)
                
                max_value = max(neighbors)
                img_out[i, j] = max_value
        
        self.image = img_out
        self.displayImage(2)

    # F1
    def SobelClicked(self):
        if self.image is None:
            return

        if len(self.image.shape) == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        Gx = cv2.filter2D(gray, cv2.CV_64F, sobel_x)
        Gy = cv2.filter2D(gray, cv2.CV_64F, sobel_y)

        gradien = np.sqrt(Gx**2 + Gy**2)

        gradien = (gradien / gradien.max()) * 255
        gradien = gradien.astype(np.uint8)

        plt.imshow(gradien, cmap='gray', interpolation='bicubic')
        plt.title('Sobel Edge Detection')
        plt.axis('off')
        plt.show()

    # F2
    def CannyEdgeClicked(self):
        if self.image is None:
            return

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gauss = (1.0 / 57) * np.array([
            [0, 1, 2, 1, 0],
            [1, 3, 5, 3, 1],
            [2, 5, 9, 5, 2],
            [1, 3, 5, 3, 1],
            [0, 1, 2, 1, 0]
        ])
        blur = cv2.filter2D(gray, -1, gauss)

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        Gx = cv2.filter2D(blur, cv2.CV_64F, sobel_x)
        Gy = cv2.filter2D(blur, cv2.CV_64F, sobel_y)

        magnitude = np.sqrt(Gx**2 + Gy**2)
        magnitude = (magnitude / magnitude.max()) * 255
        img_out = magnitude.astype(np.uint8)
        theta = np.arctan2(Gy, Gx)

        H, W = img_out.shape
        Z = np.zeros((H, W), dtype=np.uint8)
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255
                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img_out[i, j + 1]
                        r = img_out[i, j - 1]
                    # angel 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img_out[i + 1, j - 1]
                        r = img_out[i - 1, j + 1]
                    # angel 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img_out[i + 1, j]
                        r = img_out[i - 1, j]
                    # angel 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img_out[i - 1, j - 1]
                        r = img_out[i + 1, j + 1]

                    if (img_out[i, j] >= q) and (img_out[i, j] >= r):
                        Z[i, j] = img_out[i, j]
                    else:
                        Z[i, j] = 0
                except IndexError:
                    pass

        img_N = Z.astype("uint8")

        weak = 100
        strong = 150

        for i in np.arange(H):
            for j in np.arange(W):
                a = img_N[i, j]
                if a > strong:
                    img_N[i, j] = 255
                elif a > weak:
                    img_N[i, j] = weak
                else:
                    img_N[i, j] = 0
        img_H1 = img_N.astype("uint8")

        #hysteresis Thresholding eliminasi titik tepi lemah jika tidakterhubung dengan tetangga tepi kuat
        
        strong = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if img_H1[i, j] == weak:
                    try:
                        if ((img_H1[i + 1, j - 1] == strong) or
                            (img_H1[i + 1, j] == strong) or
                            (img_H1[i + 1, j + 1] == strong) or
                            (img_H1[i, j - 1] == strong) or
                            (img_H1[i, j + 1] == strong) or
                            (img_H1[i - 1, j - 1] == strong) or
                            (img_H1[i - 1, j] == strong) or
                            (img_H1[i - 1, j + 1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError:
                        pass

        img_H2 = img_H1.astype("uint8")


        cv2.imshow("Manual Canny Edge Detection", img_H2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8
        if len(self.image.shape) == 3:
            if self.image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        img = img.rgbSwapped()
        
        if windows == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        if windows == 2:
            self.prosesLabel.setPixmap(QPixmap.fromImage(img))
            self.prosesLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle('152023146 - Rizki Saepul Aziz')
    window.show()
    sys.exit(app.exec_())
