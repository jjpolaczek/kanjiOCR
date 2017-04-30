import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from controls import Controls
from ocr import OCR
import time

#MAIN
ocrEngine = OCR("log/model.ckpt", "log/dict.pickle")
#get list of source images in images folder
imageList = [join("./images/",f) for f in listdir("./images/") if isfile(join("./images/",f)) and f[0] != '.' and f != 'test.jpeg']
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
cv2.imshow('display',image)
while True:
    if currentImageNo != ctrl.t[3]: 
        image = cv2.imread(imageList[ctrl.t[3]])
        currentImageNo = ctrl.t[3]

    cv2.imshow('display',image)
    key = cv2.waitKey(10000)
    cutouts, chars = ocrEngine.ProcessImage(image)
    print ''.join(chars)
    for c in cutouts:
        #net.ProcessImage(c)
        cv2.imshow('display',c)
        key = cv2.waitKey (1000)
        if key == 27: #escape key
            key = 1
            break
    cv2.imshow('display',image)
    key = cv2.waitKey (1000)
    if key == 27: #escape key 
        break
cv2.destroyAllWindows()
for i in range(1,10):
    cv2.waitKey(1)

