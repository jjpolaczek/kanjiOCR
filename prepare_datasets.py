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

from JISX0208 import JISX0208Dict 
from JISX201 import JISX201Dict
import cv2
import re

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
              [("ETL1.zip",105028771),("ETL8B.zip",27760622 )], )
ETL1path = maybeExtract(dataRoot, "ETL1.zip")
ETL8path = maybeExtract(dataRoot, "ETL8B.zip")


#extract datasets
def excludeChars(character):
  character = character.decode('utf-8')
  #definition of all excluded characters
  s = u"・,␣’'.."
  if s.find(character) != -1:
    return True
  return False
def unpack_ETL1(sourceDir, destDir):
    d = JISX201Dict()
    onlyfiles = [f for f in os.listdir(sourceDir) if os.path.isfile(os.path.join(sourceDir, f)) and f.find("ETL1C") != -1]
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
                else:
                    skip += 1
                    acc[5] += 1
                    continue
                #exclude some characters, unimportant to the dataset
                if excludeChars(d[r[3]]):
                    skip += 1
                    continue
                iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
                iP = iF.convert('P')
                iT = iP.convert('RGB')
                fn = "{:1d}{:4d}{:2x}.png".format(r[0], r[2], r[3])
                #iP.save(fn, 'PNG', bits=4)
                enhancer = ImageEnhance.Brightness(iP)
                iE = enhancer.enhance(16)
                path = os.path.join(destDir, "%d" % ord(d[r[3]].decode('utf-8')))
                if not os.path.exists(path):
                    print path
                    os.makedirs(path)
                acc[min(int(r[5]),5)] +=1
                if r[5] == 0 and int(r[3]) != 0:               
                    iE.save(os.path.join(path,fn), 'PNG')
                skip += 1
            for c in character:
                print c
            print "Quality assesment:"
            character = []
            print(acc)

def unpack_ETL8B(sourceDir, destDir):
    d = JISX0208Dict()
    onlyfiles = [f for f in os.listdir(sourceDir) if os.path.isfile(os.path.join(sourceDir, f)) and f.find("ETL8B2") != -1]
    count = 0
    for filename in onlyfiles:
      print ("Unpacking" + filename)
      with open(os.path.join(sourceDir,filename), 'r') as f:
          f.seek(0)
          skip = 1
          acc = 6*[0]
          character = []
          while True:
              f.seek(skip * 512)
              s = f.read(512)
              if s is None or len(s) == 0:
                  break
              #2H - unsigned shord sheet no > JIS code
              #4s - char reading
              #504s data string
              if len(s) < 512:
                  print "EOF"
                  skip+=1
                  continue
              r = struct.unpack('>2H4s504s', s)
              i1 = Image.frombytes('1', (64, 63), r[3], 'raw')
              fn = 'ETL8B2_{:d}_{:s}.png'.format(count, hex(r[1])[-4:])
              iI = Image.eval(i1, lambda x: not x)
              path = os.path.join(destDir, "%d" % ord(d[r[1]].decode('utf-8')))
              if not os.path.exists(path):
                  print path
                  os.makedirs(path)
              iI.save(os.path.join(path,fn), 'PNG')
              acc[0] += 1
              count += 1
              c = d[r[1]]
              #print c
              skip +=1
          print "Quality assesment:"
          print(acc)

def Filter(image):
    image = cv2.medianBlur(image, 3)
    #image = cv2.medianBlur(image, 3)
    #image = cv2.GaussianBlur(image, (3,3), 5)
    #mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
    #                                     cv2.THRESH_BINARY_INV, 7,-15)
    ret, mask = cv2.threshold(image, 0,255, cv2.THRESH_OTSU |
                                         cv2.THRESH_BINARY_INV)
    #cv2.imshow('display', mask)
    #cv2.waitKey(100)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #mask= cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,1)
    return mask
    
def Normalize(c, oSize, invert=False):
    #center and scale letter to 75x75 pix (~ETL datasets)
    dimx = oSize[0]
    dimy = oSize[1]
    #getbounding box of the fragment
    invImg = c
    if invert:
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
    #c = cv2.GaussianBlur(c,(3,3),0)
    return c
def preproc_ETL1(sourceDir, destDir, iSize, oSize):
    image_files = os.listdir(sourceDir)
    for image in image_files:
        image_file = os.path.join(sourceDir, image)
        tmpImg = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        tmpImg = Filter(tmpImg)
        tmpImg = Normalize(tmpImg, oSize, invert=True)
        if tmpImg == None:
            print "Invalid element %s" % (image)
            continue
        
        cv2.imwrite(os.path.join(destDir,image),tmpImg)
def preproc_ETL8(sourceDir, destDir, iSize, oSize):
    image_files = os.listdir(sourceDir)
    for image in image_files:
        image_file = os.path.join(sourceDir, image)
        tmpImg = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        tmpImg = Normalize(tmpImg, oSize)
        if tmpImg == None:
            print "Invalid element %s" % (image)
            continue
        
        cv2.imwrite(os.path.join(destDir,image),tmpImg)    
def load_letter(folder, iSize, oSize):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), oSize[0], oSize[1]),
                        dtype=np.uint8)
    imgNum = 0
    errNum = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        tmpImg = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
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
def process_ETL1(sourceDir, destDir):
    letFolders = [f for f in os.listdir(sourceDir) if os.path.isfile(os.path.join(sourceDir,f)) == False]
    for f in letFolders:
        destPath = os.path.join(destDir,f)
        if not os.path.exists(destPath):
            print destPath
            os.makedirs(destPath)
        data = preproc_ETL1(os.path.join(sourceDir,f), \
                               os.path.join(destDir,f),(63,64), (75,75))
        
def process_ETL8(sourceDir, destDir):
    letFolders = [f for f in os.listdir(sourceDir) if os.path.isfile(os.path.join(sourceDir,f)) == False]
    for f in letFolders:
        destPath = os.path.join(destDir,f)
        if not os.path.exists(destPath):
            print destPath
            os.makedirs(destPath)
        data = preproc_ETL8(os.path.join(sourceDir,f), \
                               os.path.join(destDir,f),(63,64), (75,75))
        
def innerLabels(d):
    d = sorted(d)
    cnt = 0
    b = []
    #create mapping of folder int name in unicode to class count
    for it in d:
        b.append((unichr(int(it)), cnt))
        cnt += 1
    return dict(b), cnt
def maybePickle(baseDir, dataPath, force=False):
    letFolders = [f for f in os.listdir(dataPath) if os.path.isfile(os.path.join(dataPath,f)) == False]
    #find unique enumeration for data labels -dictionary map of unicode - int
    labels, labelcount = innerLabels(letFolders)
    #We pack data into two datasets - validation and training
    # each letter has a label
    totalLet = len(letFolders)
    currentLet = 1
    train_dataset = np.ndarray(shape=(0,75,75), dtype=np.uint8)
    train_labels = np.ndarray(shape =(0), dtype=np.unicode0)
    test_dataset = np.ndarray(shape=(0,75,75), dtype=np.uint8)
    test_labels = np.ndarray(shape =(0), dtype=np.unicode0)
    test_ratio = 0.1
    for f in letFolders:
        data = load_letter(os.path.join(dataPath, f),(63,64), (75,75))
        if data != None:
            print u"%s" % (unichr(int(f)))
            print("Letter %d/%d" % (currentLet,totalLet),  unichr(int(f)),f)
            setSize = data.shape[0]
            testCount = int(setSize * test_ratio)
            trainCount= setSize - testCount
            train_dataset = np.vstack((train_dataset, data[0:trainCount - 1,:,:]))
            train_labels = np.hstack((train_labels, np.array((trainCount - 1)*[unichr(int(f))])))
            test_dataset = np.vstack((test_dataset, data[trainCount:setSize,:,:]))
            test_labels = np.hstack((test_labels, np.array(testCount*[unichr(int(f))])))
            currentLet += 1
        else:
            print ("No such letter", unichr(int(f)))
            
    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    print train_dataset.shape
    print train_labels.shape
    print test_dataset.shape
    print test_labels.shape
    print ("%d unique labels" % (labelcount))
    pickle_file = os.path.join(baseDir,'ETL.pickle')
    try:
        f = open(pickle_file, 'wb')
        #TODO - save also unicode TLB int - unicode dictionary for dataset
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
            'label_map':labels,
            }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    
#unpack_ETL1(ETL1path, os.path.join(ETL1path, "data"))
unpack_ETL8B(ETL8path, os.path.join(ETL8path, "data"))
#cv2.namedWindow('display',cv2.WINDOW_NORMAL)
#process_ETL1(os.path.join(ETL1path, "data"), os.path.join(dataRoot,"data"))
process_ETL8(os.path.join(ETL8path, "data"), os.path.join(dataRoot,"data"))
maybePickle(dataRoot, os.path.join(dataRoot,"data"))
