import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from controls import Controls
import time

def PreprocessingOCR(image, ctrl, resImg):
    resImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #5 for median blur seems to be a good denoising strategy - does not work well without acceleration
    #timeStart = time.time()
    resImg = cv2.medianBlur(resImg,(ctrl.t[2] - (ctrl.t[2] + 1) %2 + 2))
    mask = cv2.adaptiveThreshold(resImg, ctrl.t[1], cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                         cv2.THRESH_BINARY, 19,3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask= cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,1)
    mask = cv2.erode(mask,kernel,3)
    
    #print (time.time() - timeStart)
    return mask
def SegmentWords(mask, ctrl):
    tmp = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
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
    print image.shape
    chars = []
    #analyze columns for splits
    invImg = np.invert(image)
    imgCols = invImg.sum(0)
    #The idea is that this split will show free spaces between chars as length of zeros
    #we find all values that are zero
    zerosX = np.asarray(np.where(imgCols == 0))
    #also compute letter widths
    widthX = np.asarray(np.where(imgCols != 0))
    #and create a list of consecutive sequences
    zerosX = consecutive(zerosX[0])
    widthX = consecutive(widthX[0])
    #now just convert to list split lines
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
    print widths
    i = 0
    while (i < len(widths)):
        if widths[i] < w_threshold:
            if i == 0:
                #beggining of sequence
                widths[i+1] += widths[i]
                spaces.pop(i+1)
                print "merge beginning"
            elif i == (len(widths) -1):
                #end of sequence
                if i == 0:
                    continue
                widths[i-1] += widths[i]
                spaces.pop(-2)
                print "merge end"
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
                print "merge middle"
            widths.pop(i)
        else:
            i += 1
        
    start = spaces[0]
    print("CUTS - ", len(spaces))
    for s in spaces:
        if start == s:
            continue
        letter = image[:,start:s]       
        chars.append(letter)
        cv2.imshow('display',letter)
        cv2.waitKey(10000)
        start = s
    return chars
            
def SplitWords(cutouts):
    shapeMargin = 0.3
    retCnt = []
    #Only horizontal text is recognized
    for c in cutouts:
        h = float(c.shape[0])
        w = float(c.shape[1])
        #if (w/h - 1.0) < shapeMargin:
        #    retCnt.append(c)
        #    continue
        words = Split(c)
        retCnt += words
def Normalize(cutouts):
    #center and scale letter to 64 x 63 pix (ETL datasets)
    return cutouts

#MAIN
#get list of source images in images folder
imageList = [join("./images/",f) for f in listdir("./images/") if isfile(join("./images/",f)) and f[0] != '.']
if len(imageList) == 0:
    raise StandardError("No files in images/ directory")
#initialize window and control panel positions with stock trackbars
cv2.namedWindow('display', cv2.WINDOW_NORMAL)
controlsList = [(0,150,255), (0,255,255), (3,5,10), (0,0,len(imageList) - 1)]
ctrl = Controls(controlsList)
cv2.resizeWindow('display', 800,600)
cv2.moveWindow('display', 100,0)
image = cv2.imread(imageList[ctrl.t[3]])
currentImageNo = ctrl.t[3]

#initialize variables
resImg = np.ones((image.shape[0], image.shape[1], 3), np.uint8)

cv2.imshow('display',image)
while True:
    key = cv2.waitKey(4000)#100ms wait in event loop
    if currentImageNo != ctrl.t[3]:
        image = cv2.imread(imageList[ctrl.t[3]])
        currentImageNo = ctrl.t[3]
    resImg = PreprocessingOCR(image, ctrl,resImg)
    resImg, contours = SegmentWords(resImg,ctrl)
    rectImg = DrawContours(resImg, contours)
    cutouts = CutoutWords(resImg, contours)
    cutouts = SplitWords(cutouts)
    cutouts = Normalize(cutouts)
    cv2.imshow('display',rectImg)
    if key == 27: #escape key 
        break
    #here all vision processing should be performed
cv2.destroyAllWindows()

