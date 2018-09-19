import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)#1*C
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue#结束当前的循序，进行下一个数的循环,break是结束循环
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j]+=(X[i,:]).T
        dW[:,y[i]]+=(-(X[i,:]).T)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW/=num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW+=reg*2*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  #W的计算：Li=1/m*求和max(0,sj-syj+1)
  #注意sj,syj都是一个值。是X.W后得到向量中的某个值。下面分类讨论
  #1、当sj-syj+1<0的时候，不会对L有所贡献
  #2、当sj-syj+1>0的时候，对于sj；是X第i行和W相乘得到的第j个，即是X第i行和W第j列相乘的结果，因此对W第j列的梯度有所贡献。由于是简单点乘，其梯度就是X第i行
  #3、当sj-syj+1>0的时候，对于syj；是X第i行和W相乘得到的第yj个，即是X第i行和W第yj列相乘的结果，因此对W第yj列的梯度有所贡献。由于是-syj，因此梯度有个-号

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train=X.shape[0]
  num_class=W.shape[1]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores=X.dot(W)
  #构造出N*C的矩阵，第i行中每个元素都表示第i个图片真实类别对应的分数
  scores_correct=scores[np.arange(num_train),y]#N*1
  #再将scores_correct每一行重复C次，得到，N*C的矩阵
  scores_correct=np.reshape(np.repeat(scores_correct,num_class),(num_train,num_class))
  temp=np.maximum(0,scores-scores_correct+1)
  temp[np.arange(num_train),y]=0#N*C，y!=sj的
  loss=1/num_train*np.sum(temp)
  loss += 0.5*reg * np.sum(W * W)
  #numpy.repeat(a, repeats, axis=Nsone) 
  #功能: 将矩阵a按照给定的axis将每个元素重复repeats次数 。axis=None时返回的是列表
  #axis=1表示按列将每一列重复repeats次，变为repeats列
  #axis=0表示按行将每一行重复repeats次，变为repeats行
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #dW初始为X.T，其中对于sj-syj+1<0的部分，由于对dW不做贡献，所以应该设置为0；
  #对于sj-syj+1>0，j!=yj的部分，X.T直接保留为梯度
  #对于sj-syj+1>0，j=yj的部分，要累加sj-syj+1>0，j!=yj的个数k。对dW做了k次贡献
  #即在temp[i]大于0时 (0<i<num_classes)，dW中的第i列(初始列为0)要加上X.T, 同时第y[i]列要减去X.T
  #max函数<0处的置为0
  temp[temp>0]=1
  temp[temp<=0]=0
  #对于syj
  row_sum=np.sum(temp,axis=1)#
  temp[np.arange(num_train),y]=-row_sum
  dW+=X.T.dot(temp)
  dW/=num_train
  dW+=reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
