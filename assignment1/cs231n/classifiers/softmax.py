import numpy as np
from random import shuffle
from past.builtins import xrange

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
  #注意数值稳定性
  num_train=X.shape[0]
  num_classes=W.shape[1]
  scores=X.dot(W)
  for i in range(num_train):
    f=scores[i,:]
    f=f-np.max(f)
    loss+=-1*np.log(np.exp(f[y[i]])/np.sum(np.exp(f)))
    
    for j in range(num_classes):
      if j==y[i]:
        dW[:,y[i]]+=(-1+np.exp(f[j])/np.sum(np.exp(f)))*X[i]
      else:
        dW[:,j]+=np.exp(f[j])/np.sum(np.exp(f))*X[i]
  #求导公式参考https://blog.csdn.net/u014485485/article/details/79503762
  
  loss/=num_train
  dW/=num_train
  loss+=0.5*reg*np.sum(W*W)
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train=X.shape[0]
  num_classes=W.shape[1]
  scores=X.dot(W)
  scores=scores-np.max(scores,axis=1,keepdims=True)#并且保持输出结果的二维特性,N*D-N*1=N*D
  p=np.exp(scores)/np.sum(np.exp(scores),axis=1,keepdims=True)#N*D/N*1=N*D
  y_trueClass=np.zeros_like(scores)
  y_trueClass[range(num_train),y]=1.0
  loss=-np.sum(y_trueClass*np.log(p))#y_trueClass*要放在外面，不然可能出现log0的错误
  dW=np.dot(X.T,p-y_trueClass)
  loss/=num_train
  dW/=num_train
  loss+=0.5*reg*np.sum(W*W)
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

