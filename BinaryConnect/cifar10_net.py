
from __future__ import print_function

import sys
import os
import time

import numpy as np
np.random.seed(1234) # for reproducibility?

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import binary_net
#sys.path.append('../../BinaryConnect/')
import compress

from pylearn2.datasets.zca_dataset import ZCA_Dataset   
from pylearn2.datasets.cifar10 import CIFAR10 
from pylearn2.utils import serial

from collections import OrderedDict

import sys, getopt

def main(argv):
   percentage_prune = ''
   pruning_type = ''

   try:
      opts, args = getopt.getopt(argv,"hp:t:",["percprune=","pruntype="])
   except getopt.GetoptError:
      print('cifar10_net.py -pp <percentage_prune> -pt <prune_type>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('cifar10_net.py -pp <percentage_prune> -pt <prune_type>')
         sys.exit()
      elif opt in ("-p", "--percprune"):
        print("hi")
        percentage_prune = arg
      elif opt in ("-t", "--pruntype"):
         pruning_type = arg
   print('Percentage Pruned is "', percentage_prune)
   print('Pruning Type is "', pruning_type)  

   return percentage_prune, pruning_type

if __name__ == "__main__":

    percentage_prune, pruning_type = main(sys.argv[1:])

    print('cnnBA_binarized_'+ str(str(percentage_prune).replace(".","")) + '_' + str(pruning_type)+'.save')

    network_type = 'binarynet'
    # BN parameters
    batch_size = 50
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")
    
    # BinaryConnect    
    binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))
    
    # Training parameters
    num_epochs = 500
    print("num_epochs = "+str(num_epochs))
    
    # Decaying LR 
    LR_start = 0.001
    print("LR_start = "+str(LR_start))
    LR_fin = 0.0000003
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    train_set_size = 45000
    print("train_set_size = "+str(train_set_size))
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('Loading CIFAR-10 dataset...')
    
    train = False
    if train == True:
        train_set = CIFAR10(which_set="train",start=0,stop = train_set_size)
        valid_set = CIFAR10(which_set="train",start=train_set_size,stop = 50000)
        test_set = CIFAR10(which_set="test")
            
        # bc01 format
        # Inputs in the range [-1,+1]
        # print("Inputs in the range [-1,+1]")
        train_set.X = np.reshape(np.subtract(np.multiply(2./255.,train_set.X),1.),(-1,3,32,32))
        valid_set.X = np.reshape(np.subtract(np.multiply(2./255.,valid_set.X),1.),(-1,3,32,32))
        test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))
        
        # flatten targets
        train_set.y = np.hstack(train_set.y)
        valid_set.y = np.hstack(valid_set.y)
        test_set.y = np.hstack(test_set.y)
        
        # Onehot the targets
        train_set.y = np.float32(np.eye(10)[train_set.y])    
        valid_set.y = np.float32(np.eye(10)[valid_set.y])
        test_set.y = np.float32(np.eye(10)[test_set.y])
        
        # for hinge loss
        train_set.y = 2* train_set.y - 1.
        valid_set.y = 2* valid_set.y - 1.
        test_set.y = 2* test_set.y - 1.

    print('Building the CNN...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    def build_model(numfilters1, numfilters2, numfilters3, numfilters4, numfilters5, numfilters6):
        cnn = lasagne.layers.InputLayer(
            shape=(None, 3, 32, 32),
            input_var=input)
    
        # 128C3-128C3-P2             
        cnn = binary_net.Conv2DLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters1, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        act1 = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation) 
                
        cnn = binary_net.Conv2DLayer(
                act1, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters2, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
        
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        act2 = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation) 
                
        # 256C3-256C3-P2             
        cnn = binary_net.Conv2DLayer(
                act2, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters3, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        act3 = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation) 
                
        cnn = binary_net.Conv2DLayer(
                act3, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters4, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
        
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        act4 = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation) 
                
        # 512C3-512C3-P2              
        cnn = binary_net.Conv2DLayer(
                act4, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters5, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        act5 = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation) 
                      
        cnn = binary_net.Conv2DLayer(
                act5, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters6, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
        
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        act6 = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation) 
        
        # print(cnn.output_shape)
        
        # 1024FP-1024FP-10FP            
        cnn = binary_net.DenseLayer(
                    act6, 
                    binary=binary,
                    stochastic=stochastic,
                    H=H,
                    W_LR_scale=W_LR_scale,
                    nonlinearity=lasagne.nonlinearities.identity,
                    num_units=1024)      
                      
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation) 
                
        cnn = binary_net.DenseLayer(
                    cnn, 
                    binary=binary,
                    stochastic=stochastic,
                    H=H,
                    W_LR_scale=W_LR_scale,
                    nonlinearity=lasagne.nonlinearities.identity,
                    num_units=1024)      
                      
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation) 
        
        cnn = binary_net.DenseLayer(
                    cnn, 
                    binary=binary,
                    stochastic=stochastic,
                    H=H,
                    W_LR_scale=W_LR_scale,
                    nonlinearity=lasagne.nonlinearities.identity,
                    num_units=10)      
                      
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)

        return cnn, act1, act2, act3, act4, act5, act6

    cnn, act1, act2, act3, act4, act5, act6 = build_model(128,128,256,256,512,512)

    

    activations = [lasagne.layers.get_output(act1), lasagne.layers.get_output(act2),lasagne.layers.get_output(act3),
    lasagne.layers.get_output(act4), lasagne.layers.get_output(act5), lasagne.layers.get_output(act6)]

    func_activations = [theano.function([input], [activations[0]]), theano.function([input], [activations[1]]),theano.function([input], [activations[2]]),
    theano.function([input], [activations[3]]), theano.function([input], [activations[4]]), theano.function([input], [activations[5]])]
    
    cnn = load_model('cnnBA_binarized.save', cnn)
    
    activations_output = []
    tmp = []
    batches = len(valid_set.X)/batch_size
    
    for j in range(len(activations)):
        for i in range(batches):
            new_act = func_activations[j](valid_set.X[i*batch_size:(i+1)*batch_size])
            tmp.append(new_act)
        activations_output.append(tmp)
        tmp = []
    
    print(len(activations_output[0]))

    print("hi")


    if train == True:
        cnn = load_model('/home/jfar0131/job3/BinaryConnect/cnnBA_binarized.save', cnn)

    params_binary = lasagne.layers.get_all_param_values(cnn, binary=True)
    params = lasagne.layers.get_all_params(cnn)
    param_values = lasagne.layers.get_all_param_values(cnn)

    if pruning_type == 'random':
        new_param_values, filter_sizes = compress.random_pruning(params_binary, param_values,float(percentage_prune), network_type)
    elif pruning_type == 'quantization':
        new_param_values, filter_sizes = compress.quantized_weights_pruning(params_binary, param_values,float(percentage_prune), network_type)
    elif pruning_type == 'real':
        new_param_values, filter_sizes = compress.real_weights_pruning(params_binary, param_values,float(percentage_prune), network_type)
    elif pruning_type == 'activation':
        new_param_values, filter_sizes = compress.activations_pruning(activations,float(percentage_prune), network_type)        

    cnn_pruned = build_model(filter_sizes[0],filter_sizes[1],filter_sizes[2],filter_sizes[3],filter_sizes[4],filter_sizes[5])

    train_output = lasagne.layers.get_output(cnn_pruned, deterministic=False)
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    if binary:
        
        # W updates
        W = lasagne.layers.get_all_params(cnn_pruned, binary=True)
        W_grads = binary_net.compute_grads(loss,cnn_pruned)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_net.clipping_scaling(updates,cnn_pruned)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(cnn_pruned, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(cnn_pruned, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(cnn_pruned, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)

    lasagne.layers.set_all_param_values(cnn_pruned, new_param_values)

    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:    
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    lasagne.layers.set_all_param_values(cnn_pruned, new_param_values)

    print('Training...')
    
    if train == True:
        binary_net.train(
                train_fn,val_fn,
                cnn_pruned, percentage_prune, pruning_type,
                batch_size,
                LR_start,LR_decay,
                num_epochs,
                train_set.X,train_set.y,
                valid_set.X,valid_set.y,
                test_set.X,test_set.y,
                shuffle_parts=shuffle_parts)
