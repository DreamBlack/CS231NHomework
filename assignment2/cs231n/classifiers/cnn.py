from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
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
        C,H,W=input_dim
        F=num_filters
        W1=weight_scale*np.random.randn(F,C,filter_size,filter_size)#别忘了乘通道数
        b1=np.zeros(F)
        #np.random.normal和np.random.randn
        #2*2不加填充和2步长的maxpool会是输入长度减半
        #但是有个问题，卷积的时候，不是也会时特征图大小发生变化么，先按下不表
        #解答：因为下面求Loss时候，设定了会让卷积之后图片大小不发生变化的pad,stride
        #上一步之后size=f*newh*neww,但是下面affine的时候不论输入时什么样的都会，把它变成n*d这种二维的再进行计算，所以w2是二维的就行,且输出也是二维的
        
        W2=weight_scale*np.random.randn(F*int(H/2)*int(W/2),hidden_dim)
        b2=np.zeros(hidden_dim)
        
        W3=weight_scale*np.random.randn(hidden_dim,num_classes)
        b3=np.zeros(num_classes)
        
        self.params['W1']=W1
        self.params['b1']=b1
        self.params['W2']=W2
        self.params['b2']=b2
        self.params['W3']=W3
        self.params['b3']=b3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}#" // " 表示整数除法,返回不大于结果的一个最大的整数

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        #conv - relu - 2x2 max pool - affine - relu - affine - softmax
        #softmax之前没有relu
        out,cache1=conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out,cache2=affine_relu_forward(out, W2, b2)
        scores,cache3=affine_forward(out, W3, b3)#!!!!!!最后一步是affine，没有relu了
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
        loss,dout=softmax_loss(scores,y)
        loss+=0.5*self.reg*(np.sum(W1**2)+np.sum(W2**2)+np.sum(W3**2))
        
        #解决loss，下面求grads
        
        dout,dW3,db3=affine_backward(dout, cache3)
        dout,dW2,db2=affine_relu_backward(dout, cache2)
        dout,dW1,db1=conv_relu_pool_backward(dout, cache1)
        
        #求梯度的时候不要忘了加上reg的部分
        dW1+=self.reg*W1
        dW2+=self.reg*W2
        dW3+=self.reg*W3
        
        grads['W1']=dW1
        grads['b1']=db1
        grads['W2']=dW2
        grads['b2']=db2
        grads['W3']=dW3
        grads['b3']=db3#b3算的不对，因为!!!!!!最后一步是affine，没有relu了
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
