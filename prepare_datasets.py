# -*- coding: utf8 -*-
import numpy as np
import os
import sys
import zipfile
import time
import struct
from PIL import Image, ImageEnhance, ImageOps
#from IPython.display import display, Image
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from JISX201 import JISX201Dict

last_percent_reported = None
def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybeDownload(dataRoot,url, datasets, force = False):
    #datasets filename create
    if not os.path.exists(dataRoot):
        os.makedirs(dataRoot)
    for filename in datasets:
        """Download a file if not present, and make sure it's the right size."""
        print url
        dest_filename = os.path.join(dataRoot, filename[0])
        if force or not os.path.exists(dest_filename):
            print('Attempting to download:', filename[0]) 
            filename, _ = urlretrieve(url + filename[0], dest_filename, reporthook=download_progress_hook)
            print('\nDownload Complete!')
            time.sleep(5)
        statinfo = os.stat(dest_filename)
        if statinfo.st_size == filename[1]:
            print('Found and verified', dest_filename)
        else:
            print(os.stat(dest_filename),filename[1])
            raise Exception(
              'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename
def maybeExtract(dataRoot, filename,force=False):
    folder = os.path.splitext(filename)[0]
    folderPath = os.path.join(dataRoot, folder)
    if os.path.isdir(folderPath) and not force:
        print('%s already present - Skipping extraction of %s.' % (folderPath, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % folderPath)
        tar = zipfile.ZipFile(os.path.join(dataRoot,filename))
        tar.extractall(dataRoot)
        tar.close()
        print ("Succesfully extracted %s)" % filename)
        print sorted(os.listdir(folderPath))
    return folderPath
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
#main aquire datasets
dataRoot = "./datasets/"
maybeDownload(dataRoot,"http://etlcdb.db.aist.go.jp/etlcdb/data/",\
              [("ETL1.zip",105028771),("ETL6.zip",165893957)], )
ETL1path = maybeExtract(dataRoot, "ETL1.zip")
ETL6path = maybeExtract(dataRoot, "ETL6.zip")


#extract datasets



def shift_jis2unicode(charcode): # charcode is an integer
    print charcode
    if charcode <= 0xFF:
        shift_jis_string = chr(charcode)
    else:
        shift_jis_string = chr(charcode >> 8) + chr(charcode & 0xFF)

    unicode_string = shift_jis_string.decode('shift-jis')
    assert len(unicode_string) == 1
    return ord(unicode_string)
def unpack_ETL1(sourceDir, destDir):
    d = JISX201Dict()
    onlyfiles = [f for f in os.listdir(sourceDir) if os.path.isfile(os.path.join(sourceDir, f)) and f.find("ETL") != -1]
    for filename in onlyfiles:
        print "Unpacking" + filename
        with open(os.path.join(sourceDir,filename), 'r') as f:
            f.seek(0)
            skip = 0
            acc = 6*[0]
            character = []
            while True:
                f.seek(skip * 2052)
                s = f.read(2052)
                if s is None or len(s) == 0:
                    break
                #H - unsigned shord
                #2s - ss -char char
                #H - insigned short
                #6B - unsigned char
                #I uint
                #4H ushort
                #4B uchar
                #4x pas bytes
                #2016s string - char
                #4x pad bytes
                if len(s) < 2052:
                    print "EOF"
                    skip+=1
                    continue
                r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
                if int(r[3]) != 0:
                    if len(character) == 0:
                        character.append(d[r[3]])
                    elif len(character) == 1 and d[r[3]] != character[0]:
                        character.append(d[r[3]])
                    elif len(character) > 1 and d[r[3]] != character[-1]:
                        character.append(d[r[3]])
                    
                iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
                iP = iF.convert('P')
                iT = iP.convert('RGB')
                fn = "{:1d}{:4d}{:2x}.png".format(r[0], r[2], r[3])
                #iP.save(fn, 'PNG', bits=4)
                enhancer = ImageEnhance.Brightness(iP)
                iE = enhancer.enhance(16)
                path = os.path.join(destDir, "%d" % (struct.unpack('H', r[1])[0]))
                if not os.path.exists(path):
                    print path
                    os.makedirs(path)
                #print r[5]
                acc[min(int(r[5]),5)] +=1
                if r[5] == 0 and int(r[3]) != 0:               
                    iE.save(os.path.join(path,fn), 'PNG')
                skip += 1
            for c in character:
                print c
            print "Quality assesment:"
            character = []
            print(acc)
            
def Filter(image):
    image = cv2.medianBlur(image, 3)
    #image = cv2.medianBlur(image, 3)
    image = cv2.GaussianBlur(image, (3,3), 10)
    #unsharp = cv2.GaussianBlur(image, (9,9), 10.0)
    #image = cv2.addWeighted(image,1.5,unsharp,-0.5,0,image)
    mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                         cv2.THRESH_BINARY_INV, 13,-15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #mask= cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,1)
    return mask
    
def Normalize(c, oSize):
    #center and scale letter to 75x75 pix (~ETL datasets)
    dimx = oSize[0]
    dimy = oSize[1]
    #getbounding box of the fragment
    invImg = np.invert(c)
    xsum = invImg.sum(0)
    ysum = invImg.sum(1)
    xBlack = np.asarray(np.where(xsum != 0))
    xBlack = xBlack[0]
    yBlack = np.asarray(np.where(ysum != 0))
    yBlack = yBlack[0]
    if xBlack.shape[0] == 0 or yBlack.shape[0] == 0:
        #print "Error- white box letter"
        #print ("SHAPE - ",c.shape)
        #print c
        return None
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
    #print(xBorder, yBorder)
    c = cv2.copyMakeBorder(c, top=yBorder, bottom=yBorder, \
                              left=xBorder, right=xBorder,\
                              borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )
    c = cv2.resize(c,((dimx),(dimy)))
    c = cv2.GaussianBlur(c,(3,3),0)
    return c
    
def load_letter(folder, iSize, oSize):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), oSize[0], oSize[1]),
                        dtype=np.uint8)
    imgNum = 0
    errNum = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        tmpImg = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        tmpImg = Filter(tmpImg)
        tmpImg = Normalize(tmpImg, oSize)
        if tmpImg == None:
            print "Invalid element %s" % (image)
            errNum +=1
            continue
        dataset[imgNum,:,:] = tmpImg
        imgNum += 1
    dataset = dataset[0:(imgNum - errNum), :,:]
    if dataset.shape[0] == 0:
        return None
    #print('Full dataset tensor:', dataset.shape)
    #print('Mean:', np.mean(dataset))
    #print('Standard deviation:', np.std(dataset))
    return dataset
        
unpack_ETL1(ETL1path, os.path.join(ETL1path, "data"))
