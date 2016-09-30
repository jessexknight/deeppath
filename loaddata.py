import view
import tables   as tab
import random   as rnd
import numpy    as np
import theano   as t
import matplotlib.pyplot as plt
from scipy.misc import imread 
from scipy.io   import loadmat
from skimage    import color, filters

def shared_type(data, name, dtype=t.config.floatX, borrow=True):
  #return t.shared(np.asarray(data, dtype=dtype), name=name, borrow=borrow)
  return np.asarray(data, dtype=dtype)

def imnorm(img,imax=None):
  if imax is None:
    imax = np.amax(img)
  return np.array(img,dtype=np.float32) / imax 

def idx_to_bin_img(img,idx,gauss=None):
  idx[:,0] = np.clip(np.round(idx[:,0]),0,img.shape[0]-1) 
  idx[:,1] = np.clip(np.round(idx[:,1]),0,img.shape[1]-1)
  for j in range(0,len(idx)):
    img[int(idx[j,1]),int(idx[j,0])] = 1
  if gauss is not None:
    img = filters.gaussian(img, gauss)
  return img

def split_tvt(data_x, data_y):
  
  def quarter_split(data, rnd_order):
    data  = data[rnd_order,:,:]
    q     = data.shape[0] // 4
    train = data[     :2*q,:,:]
    valid = data[1+2*q:3*q,:,:]
    tests = data[1+3*q:   ,:,:]
    return [train, valid, tests]
  
  # randomly shuffle
  rnd_order = np.arange(0,data_x.shape[0])
  rnd.seed(1234)
  rnd.shuffle(rnd_order)
  
  # split into sets
  [xtrain,xvalid,xtests] = quarter_split(data_x, rnd_order)
  [ytrain,yvalid,ytests] = quarter_split(data_y, rnd_order)
  
  return [xtrain, ytrain, xvalid, yvalid, xtests, ytests]  

def scores(N=100):
  
  print('Loading Data...')
  
  imgslug = 'D:\IMG\hist\CRC-HP\Detection\img#\img#.bmp'
  matslug = 'D:\IMG\hist\CRC-HP\Detection\img#\img#_detection.mat'  

  # read the data
  imgx = np.zeros((100,500,500),dtype=np.float32)
  imgy = np.zeros((100,500,500),dtype=np.float32)
  for i in range(0,N):
    # inputs
    imgrgb = imread(imgslug.replace('#',str(i+1)))
    imgx[i,:,:] = imnorm(imgrgb[:,:,0]) # R channel
    # labels
    nucidx = np.floor(loadmat(matslug.replace('#',str(i+1)))['detection'])
    imgy[i,:,:] = imnorm(idx_to_bin_img(imgy[i,:,:], nucidx, gauss=2))
    # debug
    #view.imshow(np.squeeze(imgx[i,:,:]))
    #view.imshow(np.squeeze(imgy[i,:,:,:]))
  
  # assign to sets
  [xtrain, ytrain, xvalid, yvalid, xtests, ytests] = split_tvt(imgx, imgy)
  
  print('Done')
  return [xtrain,ytrain,xvalid,yvalid,xtests,ytests] 
  