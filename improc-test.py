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
        print(im1)
        print("AEFAEAEFE")
        print(im2)
        self.assertTrue(np.isclose(im1,im2,atol=5).all())
    def test_gaussiabur55(self):
        image = np.zeros((5,5),dtype=np.float32)
        image[2][2] = 100.0
        image[0][0] = 100.0
        print("faef")
        #print(image)
        im1 = cv2.GaussianBlur(image,(5,5),1)
        im2 = improc.GaussianBlur(image,5,1)
        print(im1)
        print(im2)
        print(im1-im2)
        self.assertTrue(np.isclose(im1,im2,atol=0.01).all())
    def test_padding(self):
        image = np.zeros((5,5),dtype=np.float32)
        image[0][0] = 100.0
        image[0][1] = 40
        image[0][2] = 60
        image[1][1] = 50
        print("adding")
        print(improc.padding(image,2))
    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
