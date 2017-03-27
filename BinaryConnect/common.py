'''
This file is a common util file
'''
import theano
import theano.tensor as T
from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.elemwise import Elemwise
import lasagne

import numpy as np

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

class Round3(UnaryScalarOp):

    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()

    def grad(self, inputs, gout):
        (gz,) = gout
        return gz,

def getRound3Elem():
    # Own rounding function, that does not set the gradient to 0 like Theano's
    round3_scalar = Round3(same_out_nocomplex, name='round3')
    return Elemwise(round3_scalar)

# The neurons' activations ternarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
def binary_tanh_unit(x):
    round3 = getRound3Elem()
    return 2.*round3(hard_sigmoid(x))-1.

def binary_sigmoid_unit(x):
    round3 = getRound3Elem()
    return round3(hard_sigmoid(x))

# The binarization function
def binarization(W,H,binary=True,deterministic=False,stochastic=False,srng=None):

    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        # print("not binary")
        Wb = W

    else:

        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)

        # Stochastic BinaryConnect
        if stochastic:

            # print("stoch")
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)

        # Deterministic BinaryConnect (round to nearest)
        else:
            # print("det")
            Wb = T.round(Wb)

        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)

    return Wb

# The weights' ternarization function,
# taken directly from the BinaryConnect github repository
# (which was made available by his authors)
def ternarization(W,H,binary=True,deterministic=False,stochastic=False,srng=None):

    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        # print("not binary")
        Wb = W

    else:

        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)
        # Wb = T.clip(W/H,-1,1)

        # Stochastic BinaryConnect
        if stochastic:

            # print("stoch")
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)

        # Deterministic BinaryConnect (round to nearest)
        else:
            # print("det")
            Wb = T.switch(T.gt(Wb,0.9), 1, Wb)
            Wb = T.switch(T.lt(Wb,-0.9), -1, Wb)
            Wb = T.switch(T.ge(Wb,-0.9) & T.le(Wb,0.9), 0, Wb)

        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)

    return Wb


# This class extends the Lasagne DenseLayer to support BinaryConnect
class DenseLayer(lasagne.layers.DenseLayer):

    def __init__(self, incoming, num_units,
        binary = True, stochastic = True, H=1.,W_LR_scale="Glorot", **kwargs): #kwargs represents aguemtns for uniform function and gett_output

        self.binary = binary
        self.stochastic = stochastic

        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.H = np.float32(np.sqrt(1.5/ (num_inputs + num_units)))
            # print("H = "+str(self.H))

        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))

        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

        if self.binary:
            super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
            # add the binary tag to weights
            self.params[self.W]=set(['binary'])

        else:
            super(DenseLayer, self).__init__(incoming, num_units, **kwargs)

    def get_output_for(self, input, deterministic=False, **kwargs):

        self.Wb = ternarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        Wr = self.W
        self.W = self.Wb

        rvalue = super(DenseLayer, self).get_output_for(input, **kwargs)

        self.W = Wr

        return rvalue

# This class extends the Lasagne Conv2DLayer to support BinaryConnect
class Conv2DLayer(lasagne.layers.Conv2DLayer):

    def __init__(self, incoming, num_filters, filter_size,
        binary = True, stochastic = True, H=1.,W_LR_scale="Glorot", **kwargs):

        self.binary = binary
        self.stochastic = stochastic

        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
            # print("H = "+str(self.H))

        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units)))
            # print("W_LR_scale = "+str(self.W_LR_scale))

        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))

        if self.binary:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
            # add the binary tag to weights
            self.params[self.W]=set(['binary'])
        else:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)

    def convolve(self, input, deterministic=False, **kwargs):

        self.Wb = ternarization( self.W, self.H, self.binary, deterministic, self.stochastic, self._srng )
        Wr = self.W
        self.W = self.Wb

        rvalue = super(Conv2DLayer, self).convolve(input, **kwargs)

        self.W = Wr

        return rvalue
