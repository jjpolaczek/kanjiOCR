import numpy as np
import os
import sys
import tarfile
#from IPython.display import display, Image
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

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
        statinfo = os.stat(dest_filename)
        if statinfo.st_size == filename[1]:
            print('Found and verified', dest_filename)
        else:
            print(os.stat(dest_filename),filename[1])
            raise Exception(
              'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename


#main
dataRoot = "./datasets/"
maybeDownload(dataRoot,"http://etlcdb.db.aist.go.jp/etlcdb/data/",\
              [("ETL1.zip",105028771),("ETL6.zip",165893957)], )
