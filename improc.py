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
def GaussianBlur(image, size, sigma):
    #return ndimage.filters.gaussian_filter(image,sigma,mode='reflect')
    def blur1d(img,f):
        res = padding(img,f.shape[0]/2)
        res= signal.convolve2d(res,f,mode='valid')
        return res
    fx = cv2.getGaussianKernel(size,sigma)
    fy = np.transpose(fx)
    #print fx
    fx= np.multiply(fx,fy)
    if image.ndim == 3:
        b, g, r = image[:,:,0], image[:,:,1], image[:,:,2]
        b = blur1d(b,fx)
        g = blur1d(g,fx)
        r = blur1d(r,fx)
        return  np.dstack((b,g,r)).astype(image.dtype)
    else:
        res = blur1d(image,fx)
        return res.astype(image.dtype)
def rectangle(image, begin,end,color,thickness):
    pix = np.expand_dims(np.expand_dims(np.array(color),axis=0),axis=0)
    #print pix.shape
    image[begin[0]+1:begin[0]+thickness+1,begin[1]-1:end[1],:] = pix#vertical left
    image[end[0] + 1:end[0]+thickness +1,begin[1]-1:end[1],:] = pix#vertical right
    image[begin[0]+2:end[0]+2,begin[1]-thickness*0 - 1:begin[1]+thickness - 1] = pix #lower horizontal
    image[begin[0]+2:end[0]+2,end[1]-thickness*0-1:end[1]+thickness-1] = pix #upper horizontal
    return image
def adaptiveThreshold(image, thresh, offset, sth):
    return image,image
#To be done later
def boundingRect(contour):
    return contour



