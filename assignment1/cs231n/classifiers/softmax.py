import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
        scores_i = X[i].dot(W)
        scores_i -= np.max(scores_i)
        score_y = scores_i[y[i]]
        exp_score = np.exp(score_y) / np.sum(np.exp(scores_i)) # only need score of y[i]
        loss += - np.log(exp_score)
        
        score_p = np.exp(scores_i) / np.sum(np.exp(scores_i)) # need gradient of every score
        for j in range(num_class):
            if y[i] == j:
                dscore = score_p[j] - 1
            else:
                dscore = score_p[j]
            dW[:,j] += dscore * X[i]   
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)
  scores -= np.max(scores) # increase stability
  mask = np.arange(num_train)
  y_scores = scores[mask, y] # using [:,np.newaxis] will give (500,1) and after 
    #broadcasting, it will get a (500,500) scores, which is unwanted
  exp_score = np.exp(y_scores) / np.sum(np.exp(scores), axis=1)
  loss = -np.sum(np.log(exp_score))
  loss /= num_train
    
  scores_p = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) 
  # if keepdims=False, it will raise broadcasting error 
  # Two dimensions are compatible when
  # they are equal, or
  # one of them is 1
  scores_p[mask, y] += -1
  dW = X.T.dot(scores_p)
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  return loss, dW

