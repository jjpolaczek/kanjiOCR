import cv2
import numpy as np
from kanjiNN import KanjiNN
import improc

def PreprocessingOCR(image,resImg):
    resImg = improc.cvtColor2Gray(image)
    resImg = improc.GaussianBlur(resImg, 5,0)
    resImg,mask = improc.adaptiveThresholdMean(resImg, 255, 75,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    print kernel
    mask= cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask
def SegmentWords(mask):
    tmp = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    tmp = cv2.dilate(tmp, kernel,iterations=15)
    im2,contours,hierarchy = cv2.findContours(tmp, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return mask, contours
def DrawContours(image, cnts):
    # loop over the contours
    image_color = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for contour in cnts:
        # get rectangle bounding contour
        [x,y,w,h] = cv2.boundingRect(contour)
        if h < 80 or w < 80:
            continue
        cv2.rectangle(image_color,(x,y),(x+w,y+h),(255,0,255),2)
    return image_color
def CutoutWords(image, cnts):
    cutouts = []
    # todo merge words if split like ko character
    for contour in cnts:
        # get rectangle bounding contour
        [x,y,w,h] = cv2.boundingRect(contour)
        if h < 80 or w < 80:
            continue
        cutouts.append(image[y:(y+h),x:(x+w)])
    return cutouts

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
def Split(image):
    #print image.shape
    chars = []
    #analyze columns for splits
    invImg = np.invert(image)
    imgCols = invImg.sum(0)
    noBorderL = False if imgCols[0] == 0 else True
    noBorderR = False if imgCols[-1] == 0 else True
    #The idea is that this split will show free spaces between chars as length of zeros
    #we find all values that are zero
    zerosX = np.asarray(np.where(imgCols == 0))
    #also compute letter widths
    widthX = np.asarray(np.where(imgCols != 0))
    #and create a list of consecutive sequences
    zerosX = consecutive(zerosX[0])
    widthX = consecutive(widthX[0])
    #now just convert to list split lines
    if (len(zerosX) < 2):
        return image
    spaces = []
    widths = []
    spaceLen = []
    for z in zerosX:
        spaces.append((z[-1] + z[0]) / 2)
        spaceLen.append(z[-1] - z[0])
    for z in widthX:
        widths.append(z[-1] - z[0])
    #create an array of indices sorted by space length excluding first and last space
    spaceLen = [i[0] for i in sorted(enumerate(spaceLen), key=lambda x:x[1])\
           if i[0] != 0 and i[0] != len(spaces)-1]
    
    #to create boundingbox we can do the same to 2nd dimension [x,x,y,y]
    imgCols = invImg.sum(1)
    #modify a little so we can get bounds even on black pixels
    imgCols[0] = 0
    imgCols[-1] = 0
    zerosY = np.asarray(np.where(imgCols == 0))
    zerosY = consecutive(zerosY[0])
    boundingBox = [zerosX[0][-1], zerosX[-1][0],zerosY[0][-1], zerosY[-1][0]]
    
    h = float(boundingBox[3] - boundingBox[2])
    w = float(boundingBox[1] - boundingBox[0])
    cuts = len(spaces)
    letterNo = round(w/h)
    
    #LETTER MERGING PROCEDURE
    #If letter is below a threshold - join it to nearest one with smallest width
    # () - letter, _ - space:
    # _()_()_(_)_()_ ====> _()_()_()_()_ delete space between them and join two consecutive widths
    w_threshold = h * 0.4 #coefficient determines minimum width percentage
    #print widths
    i = 0
    # add fake borders to merge to
    if noBorderL:
        spaces.insert(0,0)
    if noBorderR:
        spaces.append(image.shape[1] - 1)
        
    while (i < len(widths)):
        #print(noBorderL, noBorderR, i, len(widths), len(spaces))
        if widths[i] < w_threshold and len(widths) > 1:
            if i == 0:
                #beggining of sequence
                widths[i+1] += widths[i]
                spaces.pop(i+1)
            elif i == (len(widths) -1):
                #end of sequence
                if i == 0:
                    continue
                widths[i-1] += widths[i]
                spaces.pop(-2)
            else:
                #somwhere in between
                if widths[i-1] < widths[i+1]:
                    #merge left
                    widths[i-1] += widths[i]
                    spaces.pop(i)
                else:
                    #merge right
                    widths[i+1] += widths[i]
                    spaces.pop(i+1)
            widths.pop(i)
        else:
            i += 1
        
    start = spaces[0]
    for s in spaces:
        if start == s:
            continue
        letter = image[:,start:s]
        chars.append(letter)
        start = s

    return chars
            
def SplitWords(cutouts):
    shapeMargin = 0.3
    retCnt = []
    #Only horizontal text is recognized
    for c in cutouts:
        words = Split(c)
        if type(words) is not list:
            words = [words]
        retCnt += words
    return retCnt
def Normalize(cutouts):
    #center and scale letter to 75x75 pix (~ETL datasets)
    dimx = 75
    dimy = 75
    retcutouts = []
    for c in cutouts:

        #getbounding box of the fragment
        invImg = np.invert(c)
        xsum = invImg.sum(0)
        ysum = invImg.sum(1)
        xBlack = np.asarray(np.where(xsum != 0))
        xBlack = xBlack[0]
        yBlack = np.asarray(np.where(ysum != 0))
        yBlack = yBlack[0]
        if xBlack.shape[0] == 0 or yBlack.shape[0] == 0:
            print "Error- white box letter"
            print ("SHAPE - ",c.shape)
            print c
            cutouts.remove(c)
            continue
        #we now can estimate the bounding box of the character
        #we wantto scale it and fit to center of whitebackground
        #c = c[xBlack[0]:xBlack[-1], yBlack[0]: yBlack[-1]]
        c = c[yBlack[0]: yBlack[-1],xBlack[0]:xBlack[-1]]
        out = np.zeros((dimx,dimy), np.uint8)
        #it would be good to preserve shape factor
        #add 10% border, preserve scale
        xBorder = 0
        yBorder = 0
        if c.shape[0] > c.shape[1]:
            yBorder = int(0.1*float(c.shape[0]))
            xBorder = ((2 * xBorder + c.shape[0]) - c.shape[1]) / 2
        else:
            xBorder = int(0.1*float(c.shape[1]))
            yBorder = ((2 * yBorder + c.shape[1]) - c.shape[0]) / 2
        
        c = cv2.copyMakeBorder(c, top=yBorder, bottom=yBorder, \
                                  left=xBorder, right=xBorder,\
                                  borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )
        c = cv2.resize(c,((dimx),(dimy)))
        c = cv2.GaussianBlur(c,(3,3),0)
        retcutouts.append(c)
    return retcutouts        

class OCR:
    def __init__(self,modelPath,dictPath):
        self.net = KanjiNN(modelPath, dictPath)
        
    def ProcessImage(self, image):
            resImg = np.ones((image.shape[0], image.shape[1], 3), np.uint8)
            resImg = PreprocessingOCR(image,resImg)
            resImg, contours = SegmentWords(resImg)
            rectImg = DrawContours(resImg, contours)
            cutouts = CutoutWords(resImg, contours)
            cutouts = SplitWords(cutouts)
            cutouts = Normalize(cutouts)
            chars = []
            for c in cutouts:
                chars.append(self.net.ProcessImage(c))
            return cutouts, chars
