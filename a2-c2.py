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
        loadUi('gui_pertemuan2.ui', self)
        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)
        self.prosesButton.clicked.connect(self.grayClicked)
        self.actionOperasi_Pencerah.triggered.connect(self.brightness)
        self.actionOperasi_Kontras.triggered.connect(self.kontras)
        self.actionPeregangan_Kontras.triggered.connect(self.peregangan_kontras)
        self.actionNegative.triggered.connect(self.negative)
        self.actionBiner_Image.triggered.connect(self.biner)
        self.actionHistogram_Grayscale.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogramClicked)
        self.actionHistogram_Equalization.triggered.connect(self.EqualHistogramClicked)
        self.actionTranslasi_Citra.triggered.connect(self.translasiCitra)
        self.action_90.triggered.connect(self.rotasi_90)
        self.action_45.triggered.connect(self.rotasi_45)
        self.action45.triggered.connect(self.rotasi45)
        self.action90.triggered.connect(self.rotasi90)
        self.action180.triggered.connect(self.rotasi180)
        self.actionZoom_In.triggered.connect(self.zoom_in)
        self.actionZoom_Out.triggered.connect(self.zoom_out)
        self.actionSkewed.triggered.connect(self.skewed)
        self.actionCrop.triggered.connect(self.cropImage)
        self.actionTambah_dan_Kurang.triggered.connect(self.aritmatika)
        self.actionAnd.triggered.connect(self.boolean)

    # A2
    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('panda.jpg')

    def loadImage(self, flname):
        self.image = cv2.imread(flname)
        self.displayImage(1)

    # A3
    @pyqtSlot()
    def grayClicked(self):
        if self.image is not None:
            self.greyscale()
    
    def greyscale(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), dtype=np.uint8)

        for i in range(H):
            for j in range(W):
                blue, green, red = self.image[i, j]
                gray[i, j] = int(0.299 * red + 0.587 * green + 0.114 * blue) 
        self.image = gray 
        self.displayImage(2)

    # A4
    def brightness(self):
        if self.image is None:
            return

        brightness = 50
        
        img = np.clip(self.image.astype(np.int16) + brightness, 0, 255).astype(np.uint8)

        self.image = img
        self.displayImage(1)
    
    # A5
    def kontras(self):
        if self.image is None:
            return

        kontras = 1.6
        
        img = np.clip(self.image.astype(np.int16) * kontras, 0, 255).astype(np.uint8)

        self.image = img
        self.displayImage(1)
    
    # A6
    def peregangan_kontras(self):
        if self.image is None:
            return

        r_min = 0
        r_max = 255
        
        img = self.image.astype(np.float32)
        img = ((img - img.min()) / (img.max() - img.min())) * (r_max - r_min)
        img = np.clip(img, r_min, r_max).astype(np.uint8)
        
        self.image = img
        self.displayImage(1)
    
    # A7
    def negative(self):
        if self.image is None:
            return
        
        max_intensity = 255
        img = max_intensity - self.image
        
        self.image = img.astype(np.uint8)
        self.displayImage(1)

    # A8
    def biner(self):
        if self.image is None:
            return

        if len(self.image.shape) == 3:
            H, W, C = self.image.shape
            binary_image = np.zeros((H, W, C), dtype=np.uint8)

            for i in range(H):
                for j in range(W):
                    blue, green, red = self.image[i, j]
                    
                    if red == 180 and green == 180 and blue == 180:
                        binary_image[i, j] = [0, 0, 0]
                    elif red < 180 and green < 180 and blue < 180:
                        binary_image[i, j] = [1, 1, 1]
                    else:
                        binary_image[i, j] = [255, 255, 255]

        else:
            H, W = self.image.shape
            binary_image = np.zeros((H, W), dtype=np.uint8)

            for i in range(H):
                for j in range(W):
                    pixel = self.image[i, j]
                    
                    if pixel == 180:
                        binary_image[i, j] = 0
                    elif pixel < 180:
                        binary_image[i, j] = 1
                    else:
                        binary_image[i, j] = 255

        self.image = binary_image
        self.displayImage(2)

    # A9
    def grayHistogram(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), dtype=np.uint8)

        for i in range(H):
            for j in range(W):
                blue, green, red = self.image[i, j]
                gray[i, j] = int(0.299 * red + 0.587 * green + 0.114 * blue) 
        
        self.image = gray
        self.displayImage(2)

        plt.hist(gray.ravel(), 255, [0, 255])
        plt.show()

    # A10
    @pyqtSlot()
    def RGBHistogramClicked(self):
        color = ('b', 'g', 'r')
        for i,col in enumerate(color):
            histo=cv2.calcHist([self.image],[i],None,[256],[0,256])
            plt.plot(histo,color=col)
            plt.xlim([0,256])
        
        plt.show()

    # A11
    @pyqtSlot()
    def EqualHistogramClicked(self):
        hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() -
        cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.image = cdf[self.image]
        self.displayImage(2)

        plt.plot(cdf_normalized, color='b')
        plt.hist(self.image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    # B1
    def translasiCitra(self):
        h,w=self.image.shape[:2]
        quarter_h,quarter_w=h/4,w/4
        T=np.float32([[1,0,quarter_w],[0,1,quarter_h]])
        img=cv2.warpAffine(self.image,T,(w,h))

        self.image = img
        self.displayImage(2)
    
    # B2
    def rotasi(self,degree):
        h, w = self.image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, .7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image = cv2.warpAffine(self.image, rotationMatrix, (h, w))
        self.image=rot_image
        self.displayImage(2)
    
    def rotasi_90(self):
        self.rotasi(-90)
    
    def rotasi_45(self):
        self.rotasi(-45)
    
    def rotasi45(self):
        self.rotasi(45)
    
    def rotasi90(self):
        self.rotasi(90)
    
    def rotasi180(self):
        self.rotasi(180)

    # B3
    def zoom_in(self):
        resize_img=cv2.resize(self.image,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        self.image = resize_img 
        self.displayImage(2)
    
    def zoom_out(self):
        resize_img=cv2.resize(self.image,None,fx=0.50, fy=0.50)
        self.image = resize_img 
        self.displayImage(2)
    
    def skewed(self):
        resize_img=cv2.resize(self.image,(900,400),interpolation=cv2.INTER_AREA)
        self.image = resize_img 
        self.displayImage(2)

    # B4
    def cropImage(self):
        if self.image is None:
            return
        
        start_x = 100
        start_y = 50
        
        end_x = 300
        end_y = 250
        
        h, w = self.image.shape[:2]
        
        start_x = max(0, min(start_x, w-1))
        start_y = max(0, min(start_y, h-1))
        end_x = max(start_x+1, min(end_x, w))
        end_y = max(start_y+1, min(end_y, h))
        
        cropped_image = self.image[start_y:end_y, start_x:end_x].copy()
        
        self.image = cropped_image
        self.displayImage(2)

    # C1
    def aritmatika(self):
        img1 = cv2.imread('1.jpg', 0)
        img2 = cv2.imread('2.jpg', 0)

        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        add_img = img1 + img2
        subtract = img1 - img2
        cv2.imshow('Image Original 1', img1)
        cv2.imshow('Image Original 2', img2)
        cv2.imshow('Image Tambah', add_img)
        cv2.imshow('Image Kurang', subtract)
    
    # C2
    def boolean(self):
        img1 = cv2.imread('1.jpg', 1)
        img2 = cv2.imread('2.jpg', 1)

        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        op_and=cv2.bitwise_and(img1,img2)
        cv2.imshow('Image Original 1', img1)
        cv2.imshow('Image Original 2', img2)
        cv2.imshow('Image Boolean', op_and)

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
