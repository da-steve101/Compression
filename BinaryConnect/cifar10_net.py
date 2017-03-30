
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
import batch_norm
#sys.path.append('../../BinaryConnect/')
import compress

from pylearn2.datasets.zca_dataset import ZCA_Dataset   
from pylearn2.datasets.cifar10 import CIFAR10 
from pylearn2.utils import serial

from collections import OrderedDict

import sys, getopt

def main(argv):
   filter_percentage_prune = ''
   filter_pruning_type = ''
   load = ''
   prune = ''
   train = ''

   try:
      opts, args = getopt.getopt(argv,"hp:t:m:c:l:",["percprune=","pruntype=", "method=", "connections=", "load="])
   except getopt.GetoptError:
      print('cifar10_net.py -p <filter_percentage_prune> -t <prune_type> -m <train_method> -c <prune_connections>, -l <load_model>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('cifar10_net.py -p <filter_percentage_prune> -t <prune_type> -m <train_method> -c <prune_connections>, -l <load_model>')
         sys.exit()
      elif opt in ("-p", "--percprune"):
        print("hi")
        filter_percentage_prune = arg
      elif opt in ("-t", "--pruntype"):
         filter_pruning_type = arg
      elif opt in ("-m", "--method"):
        train = arg
      elif opt in ("-c", "--connections"):
         prune = arg
      elif opt in ("-l", "--load"):
         load = arg    
   print('Percentage Pruned is ', filter_percentage_prune)
   print('Pruning Type is ', filter_pruning_type)
   print('Prune Connections? ', prune)
   print('Train? ', train)
   print('Load model form ', load)

   return filter_percentage_prune, filter_pruning_type, prune, train, load

if __name__ == "__main__":

    filter_percentage_prune, filter_pruning_type, prune, train, load = main(sys.argv[1:])

    if filter_percentage_prune != "":
        filter_percentage_prune = 1-float(filter_percentage_prune)

    if prune != 'True':
        prune = False
    else:
        prune = True

    if train != 'False':
        train = True
    else:
        train = False
    print(train)
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
    #activation = binary_net.HWGQ
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

    if train == True:
        print('Loading CIFAR-10 dataset...')

        train_set = CIFAR10(which_set="train",start=0,stop = train_set_size)
        valid_set = CIFAR10(which_set="train",start=train_set_size,stop = 50000)
        test_set = CIFAR10(which_set="test")
            
        # bc01 format
        # Inputs in the range [-1,+1]
        # print("Inputs in the range [-1,+1]")
        train_set.X = np.reshape(np.subtract(np.multiply(2./255.,train_set.X),1.),(-1,3,32,32))
        valid_set.X = np.reshape(np.subtract(np.multiply(2./255.,valid_set.X),1.),(-1,3,32,32))
        test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))
        validation_data = valid_set.X        
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

    def build_model(numfilters1, numfilters2, numfilters3, numfilters4, numfilters5, numfilters6, binary=binary):
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
        
        act1 = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        activ = lasagne.layers.NonlinearityLayer(
                act1,
                nonlinearity=activation) 
                
        cnn = binary_net.Conv2DLayer(
                activ, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters2, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
        
        act2 = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        cnn = lasagne.layers.NonlinearityLayer(
                act2,
                nonlinearity=activation) 
                
        # 256C3-256C3-P2             
        cnn = binary_net.Conv2DLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters3, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        act3 = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        cnn = lasagne.layers.NonlinearityLayer(
                act3,
                nonlinearity=activation) 
                
        cnn = binary_net.Conv2DLayer(
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
        
        act4 = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        cnn = lasagne.layers.NonlinearityLayer(
                act4,
                nonlinearity=activation) 
                
        # 512C3-512C3-P2              
        cnn = binary_net.Conv2DLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=numfilters5, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        act5 = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        cnn = lasagne.layers.NonlinearityLayer(
                act5,
                nonlinearity=activation) 
                      
        cnn = binary_net.Conv2DLayer(
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
        
        act6 = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        cnn = lasagne.layers.NonlinearityLayer(
                act6,
                nonlinearity=activation) 
        
        # print(cnn.output_shape)
        
        # 1024FP-1024FP-10FP            
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
        train_output = lasagne.layers.get_output(cnn, deterministic=False)
        # squared hinge loss
        loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
        
        if binary:
            
            # W updates
            W = lasagne.layers.get_all_params(cnn, binary=True)
            W_grads = binary_net.compute_grads(loss,cnn)
            updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
            updates = binary_net.clipping_scaling(updates,cnn)
            
            # other parameters updates
            params = lasagne.layers.get_all_params(cnn, trainable=True, binary=False)
            updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
            
        else:
            params = lasagne.layers.get_all_params(cnn, trainable=True)
            updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

        test_output = lasagne.layers.get_output(cnn, deterministic=True)
        test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
        test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
        # and returning the corresponding training loss:    
        train_fn = theano.function([input, target, LR], loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input, target], [test_loss, test_err])

        return cnn, act1, act2, act3, act4, act5, act6, train_fn, val_fn, activ

    def build_model_prune_and_L2(numfilters1, numfilters2, numfilters3, numfilters4, numfilters5, numfilters6, binary=binary):
        Layer1_mask = T.ftensor4('Layer1_mask')
        Layer2_mask = T.ftensor4('Layer2_mask')
        Layer3_mask = T.ftensor4('Layer3_mask')
        Layer4_mask = T.ftensor4('Layer4_mask')
        Layer5_mask = T.ftensor4('Layer5_mask')
        Layer6_mask = T.ftensor4('Layer6_mask')
        Layer7_mask = T.fmatrix('Layer7_mask')
        Layer8_mask = T.fmatrix('Layer8_mask')
        Layer9_mask = T.fmatrix('Layer9_mask')

        cnn = lasagne.layers.InputLayer(
                shape=(None, 3, 32, 32),
                input_var=input)
        
        # 128C3-128C3-P2             
        cnn = binary_net.Conv2DLayer(
                cnn, 
                Layer_mask=Layer1_mask,
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=128, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation) 
                
        cnn = binary_net.Conv2DLayer(
                cnn,
                Layer_mask=Layer2_mask,             
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=128, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
        
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation) 
                
        # 256C3-256C3-P2             
        cnn = binary_net.Conv2DLayer(
                cnn, 
                Layer_mask=Layer3_mask,                        
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=256, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation) 
                
        cnn = binary_net.Conv2DLayer(
                cnn, 
                Layer_mask=Layer4_mask,            
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=256, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
        
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation) 
                
        # 512C3-512C3-P2              
        cnn = binary_net.Conv2DLayer(
                cnn, 
                Layer_mask=Layer5_mask,           
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=512, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation) 
                      
        cnn = binary_net.Conv2DLayer(
                cnn, 
                Layer_mask=Layer6_mask,                        
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                num_filters=512, 
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
        
        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
        
        cnn = lasagne.layers.BatchNormLayer(
                cnn,
                epsilon=epsilon, 
                alpha=alpha)
                    
        cnn = lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation) 
                
        # 1024FP-1024FP-10FP            
        cnn = binary_net.DenseLayer(
                    cnn, 
                    Layer_mask=Layer7_mask,            
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
                    Layer_mask=Layer8_mask,            
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
                    Layer_mask=Layer9_mask,            
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

        train_output = lasagne.layers.get_output(cnn, deterministic=False)

        layers_NZ = lasagne.layers.get_all_layers(cnn)
        L2_Regularization = lasagne.regularization.l2(layers_NZ[1].Wb) + lasagne.regularization.l2(layers_NZ[4].Wb) + lasagne.regularization.l2(layers_NZ[8].Wb) + lasagne.regularization.l2(layers_NZ[11].Wb) + lasagne.regularization.l2(layers_NZ[15].Wb) + lasagne.regularization.l2(layers_NZ[18].Wb) + lasagne.regularization.l2(layers_NZ[22].Wb) + lasagne.regularization.l2(layers_NZ[25].Wb) + lasagne.regularization.l2(layers_NZ[28].Wb)
        HW = 0.00000000412*L2_Regularization

        # squared hinge loss
        loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output))) + HW
        
        if binary:
            
            # W updates
            W = lasagne.layers.get_all_params(cnn, binary=True)
            W_grads = binary_net.compute_grads(loss,cnn, Layer1_mask, Layer2_mask, Layer3_mask, Layer4_mask, Layer5_mask, Layer6_mask, Layer7_mask, Layer8_mask, Layer9_mask)
            updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
            updates = binary_net.clipping_scaling(updates,cnn)
            
            # other parameters updates
            params = lasagne.layers.get_all_params(cnn, trainable=True, binary=False)
            updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
            
        else:
            params = lasagne.layers.get_all_params(cnn, trainable=True)
            updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

        test_output = lasagne.layers.get_output(cnn, deterministic=True)
        test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output))) + HW
        test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
        
        # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
        # and returning the corresponding training loss:
        train_fn = theano.function([input, target, LR, Layer1_mask, Layer2_mask, Layer3_mask, Layer4_mask, Layer5_mask, Layer6_mask, Layer7_mask, Layer8_mask, Layer9_mask], loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input, target, Layer1_mask, Layer2_mask, Layer3_mask, Layer4_mask, Layer5_mask, Layer6_mask, Layer7_mask, Layer8_mask, Layer9_mask], [test_loss, test_err])
        
        return cnn, train_fn, val_fn

    cnn, act1, act2, act3, act4, act5, act6, train_fn, val_fn, activ = build_model(128,128,256,256,512,512)

    #activ_output = theano.function([input], [lasagne.layers.get_output(activ)])

    #train = False
    if load != "":
        cnn = compress.load_model(load, cnn)        
        #cnn = compress.load_model('/home/jfar0131/job3/binarizedNet/cnnBA_binarized.save', cnn)

    #get parameter values
    params_binary = lasagne.layers.get_all_param_values(cnn, binary=True)
    param_values = lasagne.layers.get_all_param_values(cnn)

    #implement kernel filter pruning if defined in the command line
    if filter_pruning_type != "":
        #Must set train == True to do activations pruning
        if filter_pruning_type != "activation":
            validation_data = None
            func_activations = None
        else:
            activations = [lasagne.layers.get_output(act1), lasagne.layers.get_output(act2),lasagne.layers.get_output(act3),lasagne.layers.get_output(act4), lasagne.layers.get_output(act5), lasagne.layers.get_output(act6)]
            func_activations = [theano.function([input], [activations[0]]), theano.function([input], [activations[1]]),theano.function([input], [activations[2]]),theano.function([input], [activations[3]]), theano.function([input], [activations[4]]), theano.function([input], [activations[5]])]
            #make train True from command line argument and uncomment below if want to test activations pruning but dont want to train
            #train = False
        new_param_values, filter_sizes = compress.kernel_filter_pruning_functionality(filter_pruning_type, params_binary, param_values, filter_percentage_prune, network_type, validation_data, func_activations, batch_size)
        cnn, act1, act2, act3, act4, act5, act6, train_fn, val_fn, activ = build_model(filter_sizes[0],filter_sizes[1],filter_sizes[2],filter_sizes[3],filter_sizes[4],filter_sizes[5])   
        lasagne.layers.set_all_param_values(cnn, new_param_values)
    #train network with or without pruning
    print(train)
    if train == True:
        print('Training...')
        if prune == False:
            binary_net.train(
                    train_fn,val_fn,
                    cnn, filter_percentage_prune, filter_pruning_type, 
                    batch_size,
                    LR_start,LR_decay,
                    num_epochs,
                    train_set.X,train_set.y,
                    valid_set.X,valid_set.y,
                    test_set.X,test_set.y,
                    shuffle_parts=shuffle_parts)
        else:
            params = lasagne.layers.get_all_param_values(cnn)
            
            count =0

            def masker(param,loops, num1, num2, num3, num4, count, thresh):
                thresh_high = thresh
                if (loops ==4):
                    mask = np.ones((num4,num3,num2,num1))
                    for k in range(num1):
                        for l in range(num2):
                            for i in range(num3):
                                for j in range(num4):
                                    if ((param[j][i][l][k] <= thresh_high) & (param[j][i][l][k] >= -thresh_high) ):
                                        mask[j][i][l][k] = 0
                                        count += 1
                                    
                else:
                    mask = np.ones((num2,num1))
                    for k in range(num1):
                        for l in range(num2):
                            if ((param[l][k] <= thresh_high) & (param[l][k] >= -thresh_high) ):
                                mask[l][k] = 0
                                count += 1
                return mask, count

            Masker1, count = masker(params[0], 4, 3, 3, 3, 128,count, 0.7)
            Masker1 = Masker1.astype(np.float32)
            Masker2, count = masker(params[6], 4, 3, 3, 128, 128,count, 0.7)
            Masker2 = Masker2.astype(np.float32)
            Masker3, count = masker(params[12], 4, 3, 3, 128, 256,count, 0.7)
            Masker3 = Masker3.astype(np.float32)
            Masker4, count = masker(params[18], 4, 3, 3, 256, 256,count, 0.7)
            Masker4 = Masker4.astype(np.float32)
            Masker5, count = masker(params[24], 4, 3, 3, 256, 512,count, 0.7)
            Masker5 = Masker5.astype(np.float32)
            Masker6, count = masker(params[30], 4, 3, 3, 512, 512,count, 0.7)
            Masker6 = Masker6.astype(np.float32)
            Masker7, count = masker(params[36], 2, 1024, 8192, 512, 512,count, 0.7)
            Masker7 = Masker7.astype(np.float32)
            Masker8, count = masker(params[42], 2, 1024, 1024, 512, 512,count,0.7)
            Masker8 = Masker8.astype(np.float32)
            Masker9, count = masker(params[48], 2, 10, 1024, 512, 512,count,0.7)
            Masker9 = Masker9.astype(np.float32)
            print(count)
            
            binary_net_prune.train(
                    train_fn,val_fn,
                    cnn,
                    batch_size,
                    LR_start,LR_decay,
                    num_epochs,
                    train_set.X,train_set.y,
                    valid_set.X,valid_set.y,
                    test_set.X,test_set.y, H,binary, Masker1, Masker2, Masker3, Masker4, Masker5, Masker6, Masker7, Masker8, Masker9,
                    shuffle_parts=shuffle_parts)
