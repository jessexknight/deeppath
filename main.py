import src
import view
import loaddata

import numpy         as np
import theano        as t
import theano.tensor as tt

import lasagne                as l
import lasagne.init           as li
import lasagne.layers         as ll
import lasagne.updates        as lu
import lasagne.objectives     as lo
import lasagne.nonlinearities as lfcn
from matplotlib.pyplot import pause

'''
-------------------------------------------------------------------------------- 
define the model
--------------------------------------------------------------------------------
'''
def build_model(input_shape, lr=0.01, mom=0.9, bs=16):
  
  def symbols():
    x = tt.tensor4('X')
    y = tt.ivector('Y')
    return x, y
  
  def define_net(input_shape, x):
    knet = ll.InputLayer     (
                              name         = 'input',
                              shape        = input_shape,
                              input_var    = x
                              )
    knet = ll.Conv2DLayer    (knet,
                              name         = 'conv1',
                              num_filters  = 16,
                              filter_size  = (9,9),
                              stride       = 1,
                              pad          = 'same',
                              nonlinearity = lfcn.rectify,
                              W            = li.GlorotUniform()
                              )
#     knet = ll.dropout        (knet,
#                               name         = 'dropout1', 
#                               p            = 0.5
#                               )
    knet = ll.Conv2DLayer    (knet,
                              name         = 'conv2',
                              num_filters  = 8,
                              filter_size  = (5,5),
                              stride       = 1,
                              pad          = 'same',
                              nonlinearity = lfcn.rectify,
                              W            = li.GlorotUniform()
                              )
    knet = ll.Conv2DLayer    (knet,
                              name         = 'predict',
                              num_filters  = 1,
                              filter_size  = (5,5),
                              stride       = 1,
                              pad          = 'same',
                              nonlinearity = lfcn.sigmoid,
                              W            = li.GlorotUniform()
                              )
    knet = ll.FlattenLayer   (knet,
                              name         = 'output',
                              outdim       = 1
                              )
    
    return knet

  def define_fcns(net, x, y):
    output     = ll.get_output(net)
    test_pred  = ll.get_output(net, deterministic=True)
    params     = ll.get_all_params(net, trainable=True)
    loss       = tt.mean(lo.squared_error(output,    y+1), dtype=t.config.floatX)
    test_loss  = tt.mean(lo.squared_error(test_pred, y+1), dtype=t.config.floatX)
    updates    = lu.nesterov_momentum(loss, params, learning_rate=lr, momentum=mom)
    train_fcn  = t.function([x,y], loss,      updates=updates, allow_input_downcast=True)
    valid_fcn  = t.function([x,y], test_loss, updates=None,    allow_input_downcast=True)
    preds_fcn  = t.function([x],   test_pred, updates=None,    allow_input_downcast=True)
   
    return [output, params, train_fcn, valid_fcn, preds_fcn]
  
  print('Defining Net...')
  x, y = symbols()
  knet = define_net(input_shape, x)
  [output, params, train_fcn, valid_fcn, preds_fcn] = define_fcns(knet, x, y)
  print('Done')
  return [knet, output, params, train_fcn, valid_fcn, preds_fcn]


'''
--------------------------------------------------------------------------------
train the model
--------------------------------------------------------------------------------
'''
def train_net(xtrain, ytrain, xvalid, yvalid,
              net, train_fcn, valid_fcn, preds_fcn,
              lr=0.01, ne=100, bs=16):
        
  def cmd_update(e, ne, terr, ntb, verr, nvb):
    print("[{}/{}]"     .format(e+1,ne))
    print("  T loss =\t{:.5}".format(terr/ntb))
    print("  V loss =\t{:.5}".format(verr/nvb))
  
  def img_update(preds_fcn, xtrain, ytrain, idx=0):
    # ypreds = make_preds(preds_fcn, xtrain, ytrain, ytrain.shape[0])
    # view.show_xyp(xtrain[idx,:,:], ytrain[idx,:,:], ypreds[idx,:,:])
    view.show_filters(net,1)
  
  print('Training Net...')
  for epoch in range(ne):
    
    # full batch training
    train_err     = 0
    train_batches = 0
    for batch in iterate_batches(xtrain, ytrain, bs):
      xb, yb = batch
      train_err += train_fcn(xb, yb.flatten())
      train_batches += 1
      
    # full batch validation
    valid_err     = 0
    valid_batches = 0
    for batch in iterate_batches(xvalid, yvalid, bs):
      xb, yb = batch
      valid_err += valid_fcn(xb, yb.flatten())
      valid_batches += 1
      
    # user updates
    cmd_update(epoch, ne, train_err, train_batches, valid_err, valid_batches)
    img_update(preds_fcn, xtrain, ytrain)
    
  print('Done')
  
  return net

'''
--------------------------------------------------------------------------------
hyperparameters
--------------------------------------------------------------------------------
'''
def define_hypers():
  # make into .cfg file
  learning_rate = 100
  n_epochs      = 10
  batch_size    = 10L
  img_size      = (500L, 500L)
  input_size    = (batch_size, 1L, img_size[0], img_size[1])
  return [learning_rate, n_epochs, batch_size, img_size, input_size]

def add_unity_dim(x,idx):
  # inserts a unity dimension at index (for 4D tensor compatibility)
  return x.reshape(src.cat(x.shape[0:idx],1L,x.shape[idx:]))

def iterate_batches(x, y, batch_size):
  for i in range(0, len(x)-batch_size+1, batch_size):
    idx = slice(i, i+batch_size)
    xi  = add_unity_dim(x[idx],1)
    yi  = y[idx]
    yield xi, yi

def make_preds(preds_fcn, x, y, batch_size):
  return np.reshape([preds_fcn(xb) for xb, _ in
                    iterate_batches(x, y, batch_size)],y.shape)

'''
--------------------------------------------------------------------------------
main
--------------------------------------------------------------------------------
'''

[xtrain,ytrain,xvalid,yvalid,xtests,ytests] = loaddata.scores()

[learning_rate, n_epochs, batch_size, img_size, input_size] = define_hypers()
[knet, output, params, train_fcn, valid_fcn, preds_fcn]     = build_model(input_size)

tnet = train_net(xtrain, ytrain, xvalid, yvalid,
                 knet, train_fcn, valid_fcn, preds_fcn,
                 lr=learning_rate, ne=n_epochs, bs=batch_size)

# ypreds = make_preds(preds_fcn, xtests, ytests, ytests.shape[0])
# for i in range(ytests.shape[0]):
#   view.show_xyp(xtests[i,:,:], ytests[i,:,:], ypreds[i,:,:])










