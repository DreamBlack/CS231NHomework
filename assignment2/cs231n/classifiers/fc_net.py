from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        w1=np.random.normal(0,weight_scale,input_dim*hidden_dim)
        w1=w1.reshape((input_dim,hidden_dim))
        self.params['W1']=w1
        w2=np.random.normal(0,weight_scale,hidden_dim*num_classes)
        w2=w2.reshape((hidden_dim,num_classes))
        self.params['W2']=w2
        #b.shape为(m,而不是(m,1)不然要出错，因为n*m+m*1是不对的，应该是n*m+m
        self.params['b1']=np.zeros(hidden_dim)
        self.params['b2']=np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1=self.params['W1']
        W2=self.params['W2']
        b1=self.params['b1']
        b2=self.params['b2']
        outaff1,cacheaff1=affine_forward(X,W1,b1)
        
        outrelu1,cachrelu1=relu_forward(outaff1)
        
        scores,cacheaff2=affine_forward(outrelu1,W2,b2)
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss,dout=softmax_loss(scores,y)
        loss+=0.5*self.reg*(np.sum(W1**2)+np.sum(W2**2))
        
        #解决loss,下面求grads
        dout,dw2,db2=affine_backward(dout,(outrelu1,W2,b2))
        
        #relu
        dout=relu_backward(dout,outaff1)
        
        #affine
        dout,dw1,db1=affine_backward(dout,(X,W1,b1))
        
        dw1+=self.reg*W1
        dw2+=self.reg*W2
        grads['W1']=dw1
        grads['W2']=dw2
        grads['b1']=db1
        grads['b2']=db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        #忘记{affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
        #如果有三层网络。其实有两个(affine-relu)*2再加一个affine+softmax
        #input-affine1--W1,relu1-affine2--W2,relu2-affine3--W3，开始的时候漏了一个，导致精确度低
        before_dim=input_dim
        for idx,val in enumerate(hidden_dims):
            W=np.random.normal(0,weight_scale,before_dim*val)
            W=W.reshape((before_dim,val))
            b=np.zeros(val)
            self.params['W'+str(idx+1)]=W
            self.params['b'+str(idx+1)]=b
            
            if self.use_batchnorm:
                self.params['gamma'+str(idx+1)]=np.ones(val)
                self.params['beta'+str(idx+1)]=np.zeros(val)
            before_dim=val
        self.params['W'+str(self.num_layers)]=np.random.randn(hidden_dims[-1],num_classes)
        self.params['b'+str(self.num_layers)]=np.random.randn(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        #声明大小固定的list，如果只用out=[]，后面赋值out[i]=a是会报错的
        outaff,cacheaff=list(range(self.num_layers+1 )),list(range(self.num_layers+1))
        outrelu,cachrelu=list(range(self.num_layers )),list(range(self.num_layers))
        outbn,cachebn=list(range(self.num_layers )),list(range(self.num_layers))
        outdropout,cachedropout=list(range(self.num_layers )),list(range(self.num_layers))
        #W,b应该从W1,b1开始
        inX=X
        for i in range(self.num_layers):
            if i==self.num_layers-1:
                
                scores,cacheaff[self.num_layers]=affine_forward(inX,self.params['W'+str(self.num_layers)],self.params['b'+str(self.num_layers)])
            else:
                outaff[i+1],cacheaff[i+1]=affine_forward(inX,self.params['W'+str(i+1)],self.params['b'+str(i+1)])
                outlast=outaff[i+1]
                #batch normalization
                if self.use_batchnorm:
                    gammai,betai=self.params['gamma'+str(i+1)],self.params['beta'+str(i+1)]
                    outbn[i+1],cachebn[i+1]=batchnorm_forward(outaff[i+1],gammai,betai,self.bn_params[i])
                    outlast=outbn[i+1]
                #relu
                #这里写错了原来是outrelu[i+1],cachrelu[i+1]=relu_forward(outbn[i+1])，只考虑了有bn的情况，没有bn时，函数参数为空会造成错误
                outlast,cachrelu[i+1]=relu_forward(outlast)
                #dropout
                if self.use_dropout:
                    outlast,cachedropout[i+1]=dropout_forward(outlast,self.dropout_param)
                    
                inX=outlast
            
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss,dout=softmax_loss(scores,y)
        for i in range(self.num_layers):
            loss+=0.5*self.reg*(np.sum(self.params['W'+str(i+1)]**2))
                
        #解决loss,下面求grads
        
        #倒着取5-1
        #这里写了好久。一是网络前向和后向所用函数顺序要搞清楚，二是各个函数forward和backward要保持的cache以及函数参数要注意
        for i in range(self.num_layers,0,-1):
            if i==self.num_layers:
                dout,dw,db=affine_backward(dout,cacheaff[i])
            else:
                if self.use_dropout:
                    dout=dropout_backward(dout,cachedropout[i])
                    
                dout=relu_backward(dout,cachrelu[i])
                
                if self.use_batchnorm:
                    dout,dgamma,dbeta=batchnorm_backward(dout,cachebn[i])
                    grads['gamma'+str(i)]=dgamma
                    grads['beta'+str(i)]=dbeta
                dout,dw,db=affine_backward(dout,cacheaff[i])
            dw+=self.reg*self.params['W'+str(i)]
                
            grads['W'+str(i)]=dw
            grads['b'+str(i)]=db
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
