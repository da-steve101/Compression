# Copyright 2015 Matthieu Courbariaux

# This file is part of BinaryConnect.

# BinaryConnect is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# BinaryConnect is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with BinaryConnect.  If not, see <http://www.gnu.org/licenses/>.

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
import cPickle

import cPickle as pickle
import gzip

import batch_norm
import binary_connect
import compress

from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.utils import serial

import matplotlib.pyplot as plt
import csv
import pylab

from collections import OrderedDict

if __name__ == "__main__":
    
    # BN parameters
    batch_size = 50
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # Training parameters
    num_epochs = 500
    print("num_epochs = "+str(num_epochs))
    
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
    
    # Decaying LR 
    LR_start = 0.003
    print("LR_start = "+str(LR_start))
    LR_fin = 0.000002
    print("LR_fin = "+str(LR_fin))
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...
    
    train_set_size = 45000
    print("train_set_size = "+str(train_set_size))
    
    print('Loading CIFAR-10 dataset...')
    train = False
    if train == True:
        preprocessor = serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/preprocessor.pkl")
        train_set = ZCA_Dataset(
            preprocessed_dataset=serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/train.pkl"), 
            preprocessor = preprocessor,
            start=0, stop = train_set_size)
        valid_set = ZCA_Dataset(
            preprocessed_dataset= serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/train.pkl"), 
            preprocessor = preprocessor,
            start=45000, stop = 50000)  
        test_set = ZCA_Dataset(
            preprocessed_dataset= serial.load("${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/test.pkl"), 
            preprocessor = preprocessor)
            
        # bc01 format
        # print train_set.X.shape
        train_set.X = train_set.X.reshape(-1,3,32,32)
        valid_set.X = valid_set.X.reshape(-1,3,32,32)
        test_set.X = test_set.X.reshape(-1,3,32,32)
        
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
        cnn = binary_connect.Conv2DLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters1, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = batch_norm.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify) 
                
        cnn = binary_connect.Conv2DLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters2, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
        
        cnn = batch_norm.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify)
                
        # 256C3-256C3-P2             
        cnn = binary_connect.Conv2DLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters3, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = batch_norm.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify)
                
        cnn = binary_connect.Conv2DLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters4, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
        
        cnn = batch_norm.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify)
                
        # 512C3-512C3-P2              
        cnn = binary_connect.Conv2DLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters5, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = batch_norm.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify)
                      
        cnn = binary_connect.Conv2DLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters6, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
        
        cnn = batch_norm.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify)
        
        # print(cnn.output_shape)
        
        # 1024FP-1024FP-10FP            
        cnn = binary_connect.DenseLayer(
                    cnn, 
                    binary=binary,
                    stochastic=stochastic,
                    H=H,
                    W_LR_scale=W_LR_scale,
                    nonlinearity=lasagne.nonlinearities.identity,
                    num_units=1024)      
                      
        cnn = batch_norm.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify)
                
        cnn = binary_connect.DenseLayer(
                    cnn, 
                    binary=binary,
                    stochastic=stochastic,
                    H=H,
                    W_LR_scale=W_LR_scale,
                    nonlinearity=lasagne.nonlinearities.identity,
                    num_units=1024)      
                      
        cnn = batch_norm.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.rectify)
        
        cnn = binary_connect.DenseLayer(
                    cnn, 
                    binary=binary,
                    stochastic=stochastic,
                    H=H,
                    W_LR_scale=W_LR_scale,
                    nonlinearity=lasagne.nonlinearities.identity,
                    num_units=10)      
                      
        cnn = batch_norm.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha,
                nonlinearity=lasagne.nonlinearities.identity)
        return cnn #, cnn1
     #load trained model

    cnn = build_model(128,128,256,256,512,512)

    #tester = lasagne.layers.get_output(cnn1, deterministic=True)
    # tester_fn = theano.function([input], tester)
    # ones = np.ones((1,3,32,32))
    

    # cnn = load_model('../../../../BinaryConnect/cnn_binarized.save', cnn)
    #cnn = load_model('/home/jfar0131/job3/BinaryConnect/cnn_binarized.save', cnn)

    params_binary = lasagne.layers.get_all_param_values(cnn, binary=True)
    params = lasagne.layers.get_all_params(cnn)
    param_values = lasagne.layers.get_all_param_values(cnn)

    #new_param_values, filter_sizes = compress.quantized_weights_pruning(params_binary, param_values, 0.65)
    #new_param_values, filter_sizes = compress.real_weights_pruning(params_binary, param_values, 0.65)
    
    new_param_values, filter_sizes = compress.random_pruning(params_binary, param_values,0.65)

    #np.set_printoptions(threshold=np.inf)

    cnn_pruned = build_model(filter_sizes[0],filter_sizes[1],filter_sizes[2],filter_sizes[3],filter_sizes[4],filter_sizes[5])

    train_output = lasagne.layers.get_output(cnn_pruned, deterministic=False)
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
    if binary:
        
        # W updates
        W = lasagne.layers.get_all_params(cnn_pruned, binary=True)
        W_grads = binary_connect.compute_grads(loss,cnn_pruned)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_connect.clipping_scaling(updates,cnn_pruned)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(cnn_pruned, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(cnn_pruned, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(cnn_pruned, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])

    lasagne.layers.set_all_param_values(cnn_pruned, new_param_values)

    if train == True:
        print('Training...')
        binary_connect.train(
                train_fn,val_fn,
                batch_size,
                LR_start,LR_decay,
                num_epochs,
                train_set.X,train_set.y,
                valid_set.X,valid_set.y,cnn_pruned,
                test_set.X,test_set.y)

    
    #plot
    do_plot = False
    if do_plot == True:
        #calculate x-axis in terms of index percentages
        xax = []
        xaxis = []

        for i in range(len(normalized)):
            for j in range(len(normalized[i])):
                x = 100*(float(j)/float(len(normalized[i])))
                xax.append(x)
            xaxis.append(xax)
            xax = []
            normalized[i] = np.array(normalized[i])

    
        plt.ylabel('filter quantized sum normalized')
        plt.xlabel('Filter')
        plt.legend()
        pylab.plot(xaxis[0],normalized[0],'.r-', label='conv1')
        pylab.plot(xaxis[1],normalized[1], '.b-', label='conv2')
        pylab.plot(xaxis[2],normalized[2], '-m.', label='conv3')
        pylab.plot(xaxis[3],normalized[3], '-g.', label='conv4')
        pylab.plot(xaxis[4],normalized[4], '-c.', label='conv5')
        pylab.plot(xaxis[5],normalized[5], '-k.', label='conv6')

        pylab.legend(loc='upper right')
        #plt.plot(x, s1, 'b-', label='hi', x, s2, 'g-')
        plt.show()




