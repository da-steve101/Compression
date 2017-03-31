
import time

from collections import OrderedDict

import numpy as np

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1')
import theano
import theano.tensor as T

import lasagne
import cPickle
<<<<<<< HEAD
=======
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

# Own rounding function, that does not set the gradient to 0 like Theano's
class Round3(UnaryScalarOp):
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()
    
    def grad(self, inputs, gout):
        (gz,) = gout
        return gz, 
        
round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)



# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1 
# during back propagation
def binary_tanh_unit(x):
    return 2*round3(hard_sigmoid(x)) - 1

    
def binary_sigmoid_unit(x):
    return round3(hard_sigmoid(x))
    
# The weights' binarization function, 
# taken directly from the BinaryConnect github repository 
# (which was made available by his authors)
def binarization(W,H,binary=True,deterministic=False,stochastic=False,srng=None):
    
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
            Wb = T.round(Wb)
        
        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)
    
    return Wb

def ternarization(W,H,binary=True,deterministic=False,stochastic=False,srng=None):

    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        # print("not binary")
        Wb = W

    else:
        Wb = T.clip(W,-1,1)

        # Stochastic Ternary
        if stochastic:

            Wb = T.cast(T.switch(T.ge(Wb, 0),srng.binomial(n=1, p=Wb, size=T.shape(Wb)),-1*(srng.binomial(n=1, p=-Wb, size=T.shape(Wb)))),theano.config.floatX)

        # Deterministic Ternary 
        else:
            Wb = T.switch(T.gt(Wb,0.8), 1, Wb)
            Wb = T.switch(T.lt(Wb,-0.8), -1, Wb)
            Wb = T.switch(T.ge(Wb,-0.8) & T.le(Wb,0.8), 0, Wb)  

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
        
        self.Wb = ternarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        Wr = self.W
        self.W = self.Wb
            
        rvalue = super(Conv2DLayer, self).convolve(input, **kwargs)
        
        self.W = Wr
        
        return rvalue
>>>>>>> master

# This function computes the gradient of the binary weights
def compute_grads( loss, network, Masks = []):
    layers = lasagne.layers.get_all_layers(network)
    grads = []
    i = 0
    for layer in layers:
        params = layer.get_params(binary=True)
        if params:
            # print(params[0].name)
            if len(Masks) > 0:
                assert len(Masks) > i, "Masks have insuffient length"
                grads.append(theano.grad(loss, wrt=layer.Wb)*Masks[i])
                i+=1
            else:
                grads.append(theano.grad(loss, wrt=layer.Wb))
    return grads

# This functions clips the weights after the parameter update
def clipping_scaling(updates,network):

    layers = lasagne.layers.get_all_layers(network)
    updates = OrderedDict(updates)

    for layer in layers:

        params = layer.get_params(binary=True)
        for param in params:
            print("W_LR_scale = "+str(layer.W_LR_scale))
            print("H = "+str(layer.H))
            updates[param] = param + layer.W_LR_scale*(updates[param] - param)
            updates[param] = T.clip(updates[param], -layer.H,layer.H)

    return updates

# A function which shuffles a dataset
def shuffle(X,y, shuffle_parts ):

    # print(len(X))

    chunk_size = len(X)/shuffle_parts
    shuffled_range = range(chunk_size)

    X_buffer = np.copy(X[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):

        np.random.shuffle(shuffled_range)

        for i in range(chunk_size):

            X_buffer[i] = X[k*chunk_size+shuffled_range[i]]
            y_buffer[i] = y[k*chunk_size+shuffled_range[i]]

        X[k*chunk_size:(k+1)*chunk_size] = X_buffer
        y[k*chunk_size:(k+1)*chunk_size] = y_buffer

    return X,y

    # shuffled_range = range(len(X))
    # np.random.shuffle(shuffled_range)

    # new_X = np.copy(X)
    # new_y = np.copy(y)

    # for i in range(len(X)):

    # new_X[i] = X[shuffled_range[i]]
    # new_y[i] = y[shuffled_range[i]]

    # return new_X,new_y

# This function trains the model a full epoch (on the whole dataset)
def train_epoch( X, y, LR, batch_size, train_fn, Layer_masks = []):

    loss = 0
    batches = len(X)/batch_size

    for i in range(batches):
        my_args = [ X[i*batch_size:(i+1)*batch_size],
                    y[i*batch_size:(i+1)*batch_size],
                    LR ] + Layer_masks
        loss += train_fn( *my_args )#, HW)

        loss/=batches

    return loss

# This function tests the model a full epoch (on the whole dataset)
def val_epoch( X, y, activ_output = None, Layer_masks = []):

    err = 0
    loss = 0
    batches = len(X)/batch_size

    for i in range(batches):
        my_args = [ X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size] ] + Layer_masks
        new_loss, new_err = val_fn( *my_args )
        if activ_output is not None and i == 0 :
            print(activ_output(X[i*batch_size:(i+1)*batch_size]))
        err += new_err
        loss += new_loss

    err = err / batches * 100
    loss /= batches

    return err, loss

# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default trainer function in Lasagne yet)
def train(train_fn,val_fn,
          model, percentage_prune,
          pruning_type,
          batch_size,
          LR_start,LR_decay,
          num_epochs,
          X_train,y_train,
          X_val,y_val,
          X_test,y_test,
          save_path=None,
          shuffle_parts=1):

    # shuffle the train set
    X_train,y_train = shuffle(X_train,y_train, shuffle_parts )
    best_val_err = 100
    best_epoch = 1
    LR = LR_start

    # We iterate over epochs:
    for epoch in range(num_epochs):

        start_time = time.time()

        train_loss = train_epoch( X_train, y_train, LR, batch_size, train_fn )
        X_train,y_train = shuffle( X_train, y_train, shuffle_parts )

        val_err, val_loss = val_epoch( X_val, y_val )

        # test if validation error went down
        if val_err <= best_val_err:

            best_val_err = val_err
            best_epoch = epoch+1

            test_err, test_loss = val_epoch( X_test, y_test)

            f = open('cnnBA_binarized_'+ str(str(percentage_prune).replace(".","")) + '_' + str(pruning_type)+'.save', 'wb')
            cPickle.dump(lasagne.layers.get_all_param_values(model), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        epoch_duration = time.time() - start_time

        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best validation error rate:    "+str(best_val_err)+"%")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%")

        # decay the LR
        LR *= LR_decay

# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default trainer function in Lasagne yet)
def train_prune(train_fn, val_fn,
                model,
                batch_size,
                LR_start,LR_decay,
                num_epochs,
                X_train,y_train,
                X_val,y_val,
                X_test,y_test, H,
                binary, Masks,
                shuffle_parts = 1 ):

    assert len(Masks) == 9, "Must be 9 masks"
    
    # shuffle the train set
    X_train,y_train = shuffle( X_train, y_train, shuffle_parts )
    best_val_err = 100
    best_epoch = 1
    LR = LR_start
    layers_NZ = lasagne.layers.get_all_layers(cnn)
    print(layers_NZ)
    magic_idxs = [ 1, 3, 6, 8, 11, 13, 16, 18, 20 ]
    
    mask_dict = dict( zip([ layers_NZ[idx].Layer_mask for idx in magic_idxs ], Masks ) )
    Nonzeros = sum( [(l_NZ_i.Wb**2).sum() for l_NZ_i in mask_dict.keys() ])
    print(Nonzeros.eval(mask_dict))

    # We iterate over epochs:
    for epoch in range(num_epochs):

        start_time = time.time()

        print(Nonzeros.eval(mask_dict))
        train_loss = train_epoch( X_train, y_train, LR, batch_size, train_fn, Masks )

        X_train,y_train = shuffle( X_train, y_train, shuffle_parts )

        val_err, val_loss = val_epoch( X_val, y_val, Layer_Masks = Masks )

        # test if validation error went down
        if val_err <= best_val_err:

            best_val_err = val_err
            best_epoch = epoch+1

            test_err, test_loss = val_epoch( X_test, y_test, Layer_Masks = Masks)

            f = open('cnn_07_pruned.save', 'wb')
            cPickle.dump(lasagne.layers.get_all_param_values(cnn), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        epoch_duration = time.time() - start_time

        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best validation error rate:    "+str(best_val_err)+"%")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%")

        # decay the LR
        LR *= LR_decay
