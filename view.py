import matplotlib.pyplot as plt
import numpy             as np
import time
import lasagne           as l
import lasagne.layers    as ll

def imshow(I):
  ax = plt.subplot(1,1,1)
  plt.imshow(I)
  ax.axis('off')
  plt.show(block=False)
  plt.draw()
  plt.pause(0.01)

def show_xyp(X, Y, P):
  ax = plt.subplot(1,3,1)
  ax.imshow(X,vmin= 0.0,vmax=1.0)
  ax.axis('off')
  ax = plt.subplot(1,3,2)
  ax.imshow(Y,vmin= 0.0,vmax=1.0)
  ax.axis('off')
  ax = plt.subplot(1,3,3)
  ax.imshow(P,vmin= 0.0,vmax=0.1)
  ax.axis('off')
  plt.show(block=False)
  plt.draw()
  plt.pause(1)

def show_filters(net, layernum=1):
  
  def grid(N):
    nx = np.ceil(np.sqrt(N))
    ny = np.floor(N/nx)
    return nx, ny
  
  # get the data 
  knettle = ll.get_all_layers(net)
  Wb = knettle[layernum].get_params()
  W  = Wb[0].get_value()
  
  # subplot layout
  nx, ny = grid(W.shape[0])
  for iy in np.arange(0,ny):
    for ix in np.arange(0,nx):
      i  = (1 + ix + (iy)*nx).astype(np.int32)
      ax = plt.subplot(ny,nx,i)
      ax.imshow(np.squeeze(W[i-1,0,:,:]))
      ax.axis('off')
  plt.show(block=False)
  plt.draw()
  plt.pause(0.01)
  

  
