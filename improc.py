import cv2
import numpy as np
from scipy import signal, ndimage, interpolate
import math
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
def Convolve2D(image, func,border='invert101',cval=0):
    res = None
    if border == 'invert101':
        res = padding(image,func.shape[0]/2)
    elif border == 'replicate':
        res = np.lib.pad(image,func.shape[0]/2,mode='edge')
    elif border == 'constant':
        res = np.lib.pad(image,func.shape[0]/2,mode='constant',constant_values=cval)
    res = signal.convolve2d(res,func,mode='valid')
    return res
def Convolve2Lin(image, func,border='invert101',cval=0):
    res = None
    func = np.squeeze(func)
    if border == 'invert101':
        res = padding(image,func.shape[0]/2)
    elif border == 'replicate':
        res = np.lib.pad(image,func.shape[0]/2,mode='edge')
    elif border == 'constant':
        res = np.lib.pad(image,func.shape[0]/2,mode='constant',constant_values=cval)
    rows = np.vsplit(res,res.shape[0])
    funct = np.transpose(func)
    rowsres = []
    for r in rows:
        rowsres.append(np.convolve(np.squeeze(r),func,mode='valid'))
    tmp = np.vstack(rowsres)
    cols = np.hsplit(tmp,tmp.shape[1])
    #print cols[0].shape
    colsres = []
    for c in cols:
        colsres.append(np.reshape(np.convolve(np.squeeze(c), func,mode='valid'),(image.shape[0],1)))
    res = np.hstack(colsres)
   #print colsres[0].shape
   # print res.shape
    #res = signal.convolve2d(res,func,mode='valid')
    return res
def GaussianBlur(image, size, sigma,border='invert101'):
    fx = cv2.getGaussianKernel(size,sigma)
    #fy = np.transpose(fx)
    #print fx
    #fx= np.multiply(fx,fy)
    if image.ndim == 3:
        b, g, r = image[:,:,0], image[:,:,1], image[:,:,2]
        b = Convolve2Lin(b,fx)
        g = Convolve2Lin(g,fx)
        r = Convolve2Lin(r,fx)
        return  np.dstack((b,g,r)).astype(image.dtype)
    else:
        res = Convolve2Lin(image,fx,border)
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
    #refImage = Convolve2D(image,nmask,border='replicate')
    refImage = Convolve2Lin(image,np.ones((windowSize),dtype=np.float32)/windowSize,border='replicate')
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

def dilate(mask,kernel,iterations=1):
    ret = np.copy(mask)
    for i in range(iterations):
        ret = Convolve2D(ret,kernel,border='replicate')
        ret = (ret > 0).astype(np.uint8) * 255
    return ret
def erode(mask,kernel,iterations=1):
    ret = np.copy(mask)
    ret = bitwise_not(ret)
    for i in range(iterations):
        ret = Convolve2D(ret,kernel,border='replicate')
        ret = (ret > 0).astype(np.uint8) * 255
    return bitwise_not(ret)
    
    return mask
def morphOpen(mask, kernel,iterations=1):
    ret = np.copy(mask)
    for i in range(iterations):
        ret = erode(ret,kernel)
        ret = dilate(ret, kernel)
    return ret
def morphClose(mask,kernel,iterations=1):
    ret = np.copy(mask)
    for i in range(iterations):
        ret = dilate(ret, kernel)
        ret = erode(ret,kernel)
    return ret
def copyMakeBorder(image,top=0,bottom=0,left=0,right=0,value=[255,255,255]):
    newShape = image.shape
    if image.ndim == 2:
        newShape = (newShape[0] + top + bottom, newShape[1] + left + right)
    elif image.ndim == 3:
        newShape = (newShape[0] + top + bottom, newShape[1] + left + right, 3)
    ret = np.zeros(newShape, dtype=image.dtype)
    if image.ndim == 2:
        ret[:,:] = value[0]
        np.copyto(ret[top:-bottom,left:-right],image)
    elif image.ndim == 3:
        ret[:,:,:] = np.array(value,dtype=np.uint8)
        np.copyto(ret[top:-bottom,left:-right,:],image)
    else:
        raise TypeError("Invalid dimensions either color or image")

    return ret
#To be done later
def boundingRect(contour):
    #this makes sense only if contour has members#
    if not(contour.shape[0] > 0 and contour.shape[1] == 1 and contour.shape[2] == 2):
        raise TypeError("Invalid dimensions of contour")
    minx = contour[0,0,0]
    maxx = minx
    miny = contour[0,0,1]
    maxy = miny
    #print contour.shape
    for cnt in range(contour.shape[0]):
        x = contour[cnt,0,0]
        y = contour[cnt,0,1]
        if x > maxx:
            maxx = x
        if x < minx:
            minx = x
        if y > maxy:
            maxy = y
        if y < miny:
            miny = y
    return (minx,miny,maxx-minx + 1, maxy-miny + 1)
def _getNeighbours(limits, pixel):
    neigList = []
    x = pixel[0]
    y = pixel[1]
    up = y + 1
    dn = y - 1
    right = x + 1
    left = x - 1
    #constrain expansion
    if dn >= 0:
        neigList.append((x,dn))
    if right < limits[0]:
        neigList.append((right,y));

    if up < limits[1]:
        neigList.append((x,up))
    if left >= 0:
        neigList.append((left,y));

    return neigList
        
def findContours(mask):
    visited = np.zeros(mask.shape, dtype=np.uint8)
    contours = []


    def IsEdgePixel(maskPassed0, pixel):
        if maskPassed0[pixel[0],pixel[1]] > 0:
            neighs = _getNeighbours(mask.shape, pixel)
            for point in neighs:
                if maskPassed0[point[0], point[1]] == 0:
                    return True
        return False
    
    def visitRegion(maskPassed, visitedPassed,pixel):
        contour = np.array([[(j,i)]],dtype=np.int32)
        visitedPassed[i,j] = 1
        def recVisit(maskPassed1,visitedPassed1, pixel, contour):
            neighs = _getNeighbours(maskPassed1.shape, pixel)
            visitedPassed1[pixel[0],pixel[1]] = 1 # mark as visitted - it is edge as we just visited it
            for point in neighs:
                if maskPassed1[point[0],point[1]] > 0 and visitedPassed1[point[0],point[1]] == 0:
                    if IsEdgePixel(maskPassed1, point):
                        # add pixel to edgecontour
                        #print ("Stacking",point)
                        contour = np.vstack((contour,np.array([[(point[1],point[0])]],dtype=np.int32)))
                        contour = recVisit(maskPassed1,visitedPassed1, point, contour) # call neighbouring edge pixel out
                    else:
                        FillInside(maskPassed1,visitedPassed1, point)
            return contour
        def FillInside(maskPassed1,visitedPassed1, pixel):
            neighs = _getNeighbours(maskPassed1.shape, pixel)
            visitedPassed1[pixel[0],pixel[1]] = 1 # mark as visitted - it is edge as we just visited it
            for point in neighs:
                if maskPassed1[point[0],point[1]] > 0 and visitedPassed1[point[0],point[1]] == 0:
                    if not IsEdgePixel(maskPassed1, point):
                        #ignore - it will be caught later
                   # else:
                        FillInside(maskPassed1,visitedPassed1, point)
            return
        
        contour = recVisit(maskPassed,visitedPassed,pixel,contour)
        return contour
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if visited[i,j] == 0:
                if mask[i,j] != 0:
                    print (i,j)
                    contours.append(visitRegion(mask,visited,(i,j)))


    # this is stupid but order in opencv list is reversed
    return contours[::-1]
def resize(image, size):
    w=image.shape[0]
    h=image.shape[1]
    w2 = size[1]
    h2 = size[0]
    temp = np.ones((w2,h2),dtype=np.uint8)
    x_ratio = float(w - 1)/float(w2)
    y_ratio = float(h-1)/float(h2)
    for i in range(h2):
        for j in range(w2):
            x = int(x_ratio * j)
            y = int(y_ratio * i)
            x_diff = float(x_ratio * j) - x
            y_diff = float(y_ratio * i) - y
            index = y * w + x
            A = image[x,y]
            B = image[x+1,y]
            C = image[x,y+1]
            D = image[x+1,y+1]
            gray = (int)( \
                    A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) + \
                    C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff) \
                    )
            temp[j,i] = gray
    return temp


def testarea():
    image =cv2.imread("images/test.jpeg")
    cv2.namedWindow('display', cv2.WINDOW_NORMAL)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = GaussianBlur(image,5,1)

    cv2.imshow('display',image)
    while True:
        key = cv2.waitKey (100)
        if key == 27: #escape key 
            break
    cv2.destroyAllWindows()

#testarea()
