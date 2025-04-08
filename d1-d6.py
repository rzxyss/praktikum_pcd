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
        loadUi('gui_pertemuan3.ui', self)
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
