import cv2
import numpy as np
from scipy import signal, ndimage
def bitwise_not(mask):
    return np.invert(np.squeeze(mask))
def cvtColor2Gray(rgb):
    b, g, r = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.uint8)
#def MakeBorder(image, bordersize):
#numpy.pad(image,((4,4),(3,3),(0,0),'edge')
def padding_func(vector, pad_width, iaxis, kwargs):
    np.copyto(vector[:pad_width[0]] , np.flip(vector[pad_width[0]+1:pad_width[0] * 2 + 1],0))
    np.copyto(vector[-pad_width[1]:], np.flip(vector[-pad_width[1] * 2-1:-pad_width[1]-1],0))
def padding(image, pad_size):
    return np.lib.pad(image,pad_size,padding_func)
def Convolve2D(image, func,border='invert101'):
    res = None
    if border == 'invert101':
        res = padding(image,func.shape[0]/2)
    elif border == 'replicate':
        res = np.lib.pad(image,func.shape[0]/2,mode='edge')
    res = signal.convolve2d(res,func,mode='valid')
    return res
def GaussianBlur(image, size, sigma,border='invert101'):
    fx = cv2.getGaussianKernel(size,sigma)
    fy = np.transpose(fx)
    #print fx
    fx= np.multiply(fx,fy)
    if image.ndim == 3:
        b, g, r = image[:,:,0], image[:,:,1], image[:,:,2]
        b = Convolve2D(b,fx)
        g = Convolve2D(g,fx)
        r = Convolve2D(r,fx)
        return  np.dstack((b,g,r)).astype(image.dtype)
    else:
        res = Convolve2D(image,fx,border)
        return res.astype(image.dtype)
def rectangle(image, begin,end,color,thickness):
    pix = np.expand_dims(np.expand_dims(np.array(color),axis=0),axis=0)
    #print pix.shape
    image[begin[1]:begin[1]+thickness,begin[0]:end[0]+1,:] = pix#vertical left
    image[end[1]:end[1]+thickness,begin[0]:end[0]+1,:] = pix#vertical right
    image[begin[1]+1:end[1],begin[0]:begin[0]+thickness] = pix #lower horizontal
    image[begin[1]+1:end[1],end[0]:end[0]+thickness] = pix #upper horizontal
    return image
def adaptiveThresholdGaussian(image, value, windowSize, offset,sigma=0.0):
    if image.ndim != 2:
        raise TypeError('Invalid image dimentions - need grayscale')
    refImage = np.copy(image)
    refImage = GaussianBlur(refImage, windowSize, sigma,border='replicate')
    refImage = refImage - offset
    out = np.zeros(image.shape, dtype=np.uint8)
    it = np.nditer([image,refImage,out],[],
                   [['readonly'], ['readonly'], ['writeonly']])                   

    for (a,b,c) in it:
        if a > b:
            c[()] = value
        else:
            c[()] = 0
    return image,out
def adaptiveThresholdMean(image,value,windowSize,offset):
    if image.ndim != 2:
        raise TypeError('Invalid image dimentions - need grayscale')
    refImage = np.copy(image)
    nmask = np.ones((windowSize,windowSize), dtype=np.float32)
    nmask = nmask / nmask.size
    refImage = Convolve2D(image,nmask,border='replicate')
    refImage = refImage - offset
    out = np.zeros(image.shape, dtype=np.uint8)
    it = np.nditer([image,refImage,out],[],
                   [['readonly'], ['readonly'], ['writeonly']])                   

    for (a,b,c) in it:
        if a > b:
            c[()] = value
        else:
            c[()] = 0
    return image,out
#To be done later
def boundingRect(contour):
    return contour

def testarea():
    image =cv2.imread("images/test.jpeg")
    cv2.namedWindow('display', cv2.WINDOW_NORMAL)
    cv2.namedWindow('display2', cv2.WINDOW_NORMAL)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image, mask = adaptiveThresholdMean(image,255,75,10)
    mask2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                             cv2.THRESH_BINARY, 75,10)
    cv2.imshow('display2',mask2)
    cv2.imshow('display',mask)
    while True:
        key = cv2.waitKey (100)
        if key == 27: #escape key 
            break
    cv2.destroyAllWindows()
#testarea()
