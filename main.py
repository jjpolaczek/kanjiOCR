import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from controls import Controls

def PreprocessingOCR(image, ctrl, resImg):
    resImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(resImg, ctrl.t[1], cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                         cv2.THRESH_BINARY, 19,3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask= cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,1)
    mask = cv2.erode(mask,kernel,3)
    return mask
def SegmentWords(mask, ctrl):
    tmp = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    tmp = cv2.dilate(tmp, kernel,iterations=15)
    im2,contours,hierarchy = cv2.findContours(tmp, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return mask, contours
def DrawContours(image, cnts):
    # loop over the contours
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    #image = cv2.drawContours(image, cnts, -1, (0, 255,0),3)
    for contour in cnts:
        # get rectangle bounding contour
        [x,y,w,h] = cv2.boundingRect(contour)
        if h < 80 or w < 80:
            continue
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
    return image
#get list of source images in images folder
imageList = [join("./images/",f) for f in listdir("./images/") if isfile(join("./images/",f))]
if len(imageList) == 0:
    raise StandardError("No files in images/ directory")
#initialize window and control panel positions with stock trackbars
cv2.namedWindow('display', cv2.WINDOW_NORMAL)
controlsList = [(0,150,255), (0,255,255), (3,3,10), (0,0,len(imageList) - 1)]
ctrl = Controls(controlsList)
cv2.resizeWindow('display', 800,600)
cv2.moveWindow('display', 100,0)
image = cv2.imread(imageList[ctrl.t[3]])
currentImageNo = ctrl.t[3]

#initialize variables
resImg = np.ones((image.shape[0], image.shape[1], 3), np.uint8)

cv2.imshow('display',image)
while True:
    key = cv2.waitKey(100)#100ms wait in event loop
    if currentImageNo != ctrl.t[3]:
        image = cv2.imread(imageList[ctrl.t[3]])
        currentImageNo = ctrl.t[3]
    resImg = PreprocessingOCR(image, ctrl,resImg)
    resImg, contours = SegmentWords(resImg,ctrl)
    resImg = DrawContours(resImg, contours)
    cv2.imshow('display',resImg)
    if key == 27: #escape key 
        break
    #here all vision processing should be performed
cv2.destroyAllWindows()

