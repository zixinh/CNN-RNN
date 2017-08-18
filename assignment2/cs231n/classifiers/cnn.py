import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  [conv-relu-pool]XN - [affine]XM - [softmax or SVM]  
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_size=[7],
               hidden_dim=[100], num_classes=10, weight_scale=1e-3, reg=0.0, 
               use_batchnorm = False, dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.num_convlayer = len(num_filters)
    self.num_fc = len(hidden_dim)
    self.num_layers = self.num_convlayer + self.num_fc + 1
    num_filters.insert(0,input_dim[0])
    fc_input_size = input_dim[0] * input_dim[2] * input_dim[1]
    
    for i in range(1,self.num_convlayer+1):
        self.params['W' + str(i)] = weight_scale * np.random.randn(num_filters[i], 
                                                                   num_filters[i-1], filter_size[i-1], filter_size[i-1])
        self.params['b' + str(i)] = np.zeros((num_filters[i],))
        fc_input_size = fc_input_size / num_filters[i-1] * num_filters[i] / 4
        if use_batchnorm:
            self.params['gamma' + str(i)] = np.ones(num_filters[i])
            self.params['beta' + str(i)] = np.zeros(num_filters[i])
    
    hidden_dim.insert(0,fc_input_size)
    for i in range(self.num_fc):
        num_param = self.num_convlayer + 1 + i
        self.params['W' + str(num_param)] = weight_scale * np.random.randn(hidden_dim[i], hidden_dim[i+1])
        self.params['b' + str(num_param)] = np.zeros((hidden_dim[i + 1]),)
        if self.use_batchnorm:
            self.params['gamma' + str(num_param)] = np.ones(hidden_dim[i + 1])
            self.params['beta' + str(num_param)] = np.zeros(hidden_dim[i + 1])
        
    self.params['W' + str(self.num_layers)] = weight_scale * np.random.randn(hidden_dim[-1], num_classes)
    self.params['b' + str(self.num_layers)] = np.zeros((num_classes,))
    
    self.bn_params = []
    if self.use_batchnorm:
        self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)] # a list of dict
    
    del num_filters[0]
    del hidden_dim[0]
    
    #self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
    #self.params['b1'] = weight_scale * np.zeros((num_filters,))
    
    #input_h1 = input_dim[1] * input_dim[2] * num_filters / 4
    #self.params['W2'] = weight_scale * np.random.randn(input_h1, hidden_dim)
    #self.params['b2'] = weight_scale * np.zeros((hidden_dim,))
    
    #self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    #self.params['b3'] = weight_scale * np.zeros((num_classes,))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'

    conv_param = {}
    for i in range(self.num_convlayer):
        filter_size = self.params['W' + str(i+1)].shape[2]
        conv_param[i + 1] = {'stride': 1, 'pad': (filter_size - 1) / 2}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    if self.use_batchnorm:
        for bn_param in self.bn_params:
            bn_param[mode] = mode

    #W1, b1 = self.params['W1'], self.params['b1']
    #W2, b2 = self.params['W2'], self.params['b2']
    #W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    #filter_size = W1.shape[2]
    #conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    cache = []
    scores = X
    reg_sum = 0
    
    for i in range(self.num_convlayer):
        w = self.params['W'+ str(i + 1)]
        b = self.params['b' + str(i + 1)]
        if self.use_batchnorm:
            gamma = self.params['gamma' + str(i + 1)]
            beta = self.params['beta' + str(i + 1)]
            scores, cache1 = conv_batchnorm_relu_pool_forward(scores, w, b, gamma, beta, conv_param[i+1], 
                                                              pool_param, self.bn_params[i])
            #print cache1[1][1].shape

        else:
            scores, cache1 = conv_relu_pool_forward(scores, w, b, conv_param[i + 1], pool_param)
        cache.append(cache1)
        reg_sum += 0.5 * self.reg * np.sum(w ** 2)

        
    for i in range(self.num_fc):      
        cur_layer = self.num_convlayer + 1 + i
        w = self.params['W' + str(cur_layer)]
        b = self.params['b' + str(cur_layer)]
        if self.use_batchnorm:
            gamma = self.params['gamma' + str(cur_layer)]
            beta = self.params['beta' + str(cur_layer)]
            scores, cache2 = affine_batch_relu_forward(scores, w, b, gamma, beta, self.bn_params[cur_layer - 1])
        else:
            scores, cache2 = affine_relu_forward(scores, w, b)
        cache.append(cache2)
        reg_sum += 0.5 * self.reg * np.sum(w ** 2)

    cur_layer = self.num_layers
    w = self.params['W' + str(cur_layer)]
    b = self.params['b' + str(cur_layer)]
    scores, cache3 = affine_forward(scores, w, b)
    cache.append(cache3)
    reg_sum += 0.5 * self.reg * np.sum(w ** 2)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += reg_sum
    
    dscores, grads['W' + str(cur_layer)], grads['b' + str(cur_layer)] = affine_backward(dscores, cache.pop())
    #print 'W' + str(cur_layer)
    #print dscores.shape
    for i in range(self.num_fc, 0, -1):
        cur_layer = self.num_convlayer + i
        #print 'W' + str(cur_layer)
        if self.use_batchnorm:
            dscores, grads['W'+str(cur_layer)], grads['b'+str(cur_layer)], grads['gamma'+str(cur_layer)], grads['beta'+str(cur_layer)] = affine_batch_relu_backward(dscores, cache.pop())
            #print dscores.shape
        else:
            dscores, grads['W' + str(cur_layer)], grads['b' + str(cur_layer)] = affine_relu_backward(dscores, cache.pop())

    for i in range(self.num_convlayer,0, -1):
        if self.use_batchnorm:
            dscores, grads['W'+str(i)], grads['b'+str(i)], grads['gamma'+str(i)], grads['beta'+str(i)] = conv_batchnorm_relu_pool_backward(dscores, cache.pop())
        else:
            dscores, grads['W' + str(i)], grads['b' + str(i)] = conv_relu_pool_backward(dscores, cache.pop())
        #print 'W' + str(i)

    # add regularization term
    for i in range(1,self.num_layers + 1):
        loss += 0.5 * self.reg * np.sum(self.params['W' + str(i)]**2)
        grads['W' + str(i)] += self.reg * self.params['W' + str(i)]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  

