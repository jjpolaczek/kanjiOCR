import cv2
import numpy as np
import improc
import unittest

class TestImprocMethods(unittest.TestCase):

    def test_bitwisenot(self):
        image = np.random.randint(0, high=2, size=(200,200,1), dtype=np.uint8)
        image = image * 255
        self.assertTrue((cv2.bitwise_not(image)==improc.bitwise_not(image)).all())
        

    def test_color2gray(self):
        image = cv2.imread("images/test.jpeg")
        self.assertTrue(np.isclose(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),improc.cvtColor2Gray(image),atol=1).all())
    def test_gaussianfilter(self):
        image = cv2.imread("images/test.jpeg")
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        im1 = improc.GaussianBlur(image,5,1)
        im2 = cv2.GaussianBlur(image,(5,5),1)
        self.assertTrue(np.isclose(im1,im2,atol=5).all())
    def test_gaussiabur55(self):
        image = np.zeros((5,5),dtype=np.float32)
        image[2][2] = 100.0
        image[0][0] = 100.0
        im1 = cv2.GaussianBlur(image,(5,5),1)
        im2 = improc.GaussianBlur(image,5,1)
        self.assertTrue(np.isclose(im1,im2,atol=0.01).all())
    def test_padding(self):
        image = np.zeros((5,5),dtype=np.float32)
        image[0][0] = 100.0
        image[0][1] = 40
        image[0][2] = 60
        image[1][1] = 50
    def test_rectangle(self):
        image = cv2.imread("images/test.jpeg")
        x = image.shape[0] /2
        y = image.shape[1] / 2
        w = 15
        h = 20
        im1 = cv2.rectangle(np.copy(image),(x,y),(x+w,y+h),(255,0,255),1)
        im2 = improc.rectangle(np.copy(image),(x,y),(x+w,y+h),(255,0,255),1)
        self.assertTrue((im1 == im2).all())
    def test_rectangle2(self):
        image = np.zeros((15,15,3), dtype=np.uint8)
        x = 2
        y = 3
        w = 5
        h = 7
        l = 1
        im1 = cv2.rectangle(np.copy(image),(x,y),(x+w,y+h),(255,0,255),l)
        im2 = improc.rectangle(np.copy(image),(x,y),(x+w,y+h),(255,0,255),l)
        self.assertTrue((im1 == im2).all())
    def test_adaptivethresholdmean(self):
        image = cv2.imread("images/test.jpeg")
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image, mask1 = improc.adaptiveThresholdMean(image,255,75,10)
        mask2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                             cv2.THRESH_BINARY, 75,10)
        print np.sum(mask1 != mask2)
        self.assertTrue(np.sum(mask1 != mask2) < mask1.size / 100)
    def test_adaptivethresholdgauss(self):
        image = cv2.imread("images/test.jpeg")
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image, mask1 = improc.adaptiveThresholdGaussian(image,255,75,10)
        mask2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                             cv2.THRESH_BINARY, 75,10)
        print np.sum(mask1 != mask2)
        self.assertTrue(np.sum(mask1 != mask2) < mask1.size / 100)
    def test_dilate(self):
        image = np.random.randint(0,high=2,size=(15,15),dtype=np.uint8)
        image[3:13,3:13]=0
        image = image * 255
        kernel = np.ones((3,3))
        im1 = improc.dilate(image,kernel,iterations=4)
        im2 = cv2.dilate(image, kernel,iterations=4)
        self.assertTrue((im1 == im2).all())
    def test_erode(self):
        image = np.random.randint(0,high=2,size=(15,15),dtype=np.uint8)
        image[3:13,3:13]=1
        image = image *255
        kernel = np.ones((3,3))
        im1 = improc.erode(image,kernel,iterations=4)
        im2 = cv2.erode(image, kernel,iterations=4)
        self.assertTrue((im1 == im2).all())
    def test_morphopen(self):
        image = np.random.randint(0,high=2,size=(15,15),dtype=np.uint8)
        image[3:13,3:13]=1
        image = image *255
        kernel = np.ones((3,3))
        im1 = improc.morphOpen(image,kernel)
        im2 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        self.assertTrue((im1 == im2).all()) 
    def test_morphclose(self):
        image = np.random.randint(0,high=2,size=(15,15),dtype=np.uint8)
        image[3:13,3:13]=1
        image = image *255
        kernel = np.ones((3,3))
        im1 = improc.morphClose(image,kernel)
        im2 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        self.assertTrue((im1 == im2).all())
    def test_copymakeborder3D(self):
        image = np.ones((10,10,3),dtype=np.uint8)
        image = image *111
        im1 = improc.copyMakeBorder(np.copy(image),top=1,bottom=2,left=3,right=4,\
                                    value=[255,222,255])
        im2 = cv2.copyMakeBorder(np.copy(image),top=1,bottom=2,left=3,right=4, \
                                 borderType= cv2.BORDER_CONSTANT,value=[255,222,255])
        self.assertTrue((im1 == im2).all())
    def test_copymakeborder1D(self):
        image = np.ones((10,10),dtype=np.uint8)
        image = image *111
        im1 = improc.copyMakeBorder(np.copy(image),top=1,bottom=2,left=3,right=4,\
                                    value=[255,222,255])
        im2 = cv2.copyMakeBorder(np.copy(image),top=1,bottom=2,left=3,right=4, \
                                 borderType= cv2.BORDER_CONSTANT,value=[255,222,255])
        
        self.assertTrue((im1 == im2).all())
    def test_findcontours(self):
        image = np.zeros((15,15),dtype=np.uint8)
        image[3:7,3:5]=1#first rectangle
        image[7:14,8:12]=1#second rectangle
        image = image *255
        im2,res1,hierarchy = cv2.findContours(np.copy(image), cv2.RETR_EXTERNAL\
                                                  ,cv2.CHAIN_APPROX_NONE)
        res2 = improc.findContours(image)
        
        self.assertTrue(len(res1) == len(res2))
        for i in range(len(res1)):
            self.assertTrue((res1[i]==res2[i]).all())
    def test_boundingrect(self):
        image = np.zeros((15,15),dtype=np.uint8)
        image[3:7,3:4]=1
        image[7:14,8:12]=1
        image = image *255
        im2,contours,hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL\
                                                  ,cv2.CHAIN_APPROX_NONE)
        
        for cnt in contours:
            ret1 = cv2.boundingRect(cnt)
            ret2 = improc.boundingRect(cnt)
            self.assertTrue((ret1 == ret2))
    def test_resize(self):
        image = np.random.randint(0,high=2,size=(3,2),dtype=np.uint8)
        image = image *255
        dimx = 4
        dimy = 5
        res1 = cv2.resize(np.copy(image),((dimx),(dimy)))
        
        
        res2 = improc.resize(np.copy(image),((dimx),(dimy)))
        print image
        print res2
        print res1
        self.assertTrue(np.isclose(res1,res2,atol=0.01).all())

if __name__ == '__main__':
    unittest.main()
