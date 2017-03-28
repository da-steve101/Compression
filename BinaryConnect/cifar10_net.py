
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
import common
import lasagne

import cPickle as pickle
import gzip

import binary_net
import batch_norm
import cifar_model
#sys.path.append('../../BinaryConnect/')
import compress

from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.utils import serial

from collections import OrderedDict

import sys
import argparse

def main(argv):

   parser = argparse.ArgumentParser(description='Train a TNN on the cifar10 dataset')
   parser.add_argument("percprune", type=float )
   parser.add_argument("--train", action="store_true")
   parser.add_argument("--prune", action="store_true")
   parser.add_argument("--pruntype", nargs="?", choices=['random', 'quantization', 'real', 'activation', ''], default = "")
   parser.add_argument("--load", nargs="?")
   args = parser.parse_args()

   print(args)

   if args.load is not None and args.train:
      parser.error("load requires train to be false")

   print('Percentage Pruned is ', args.percprune)
   print('Pruning Type is ', args.pruntype)
   print('Prune Connections? ', args.prune)
   print('Train? ', args.train)
   print('Load model from ', args.load)

   return (1 - args.percprune), args.pruntype, args.prune, args.train, args.load

if __name__ == "__main__":

    filter_percentage_prune, filter_pruning_type, prune, train, load = main(sys.argv[1:])

    inputs = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)
    filter_sizes = [128,128,256,256,512,512]

    my_model = cifar_model.Cifar_Model( inputs, target, filter_sizes )
    my_model.LR = LR

    print(train)
    network_type = 'binarynet'
    # BN parameters
    batch_size = 50
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    my_model.alpha = .1
    print("alpha = "+str(my_model.alpha))
    my_model.epsilon = 1e-4
    print("epsilon = "+str(my_model.epsilon))

    # BinaryOut
    #activation_fn = binary_net.HWGQ
    my_model.activation_fn = common.binary_tanh_unit
    print("activation_fn = common.binary_tanh_unit")
    # activation_fn = common.binary_sigmoid_unit
    # print("activation_fn = common.binary_sigmoid_unit")

    # BinaryConnect
    my_model.binary = True
    print("binary = "+str(my_model.binary))
    my_model.stochastic = False
    print("stochastic = "+str(my_model.stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    my_model.H = 1.
    print("H = "+str(my_model.H))
    # W_LR_scale = 1.
    my_model.W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(my_model.W_LR_scale))

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
    cnn, activations, train_fn, val_fn, activ = my_model.build_model()

    activ_output = theano.function([inputs], [lasagne.layers.get_output(activ)])

    #train = False
    if load is not None:
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
            activations = [ lasagne.layers.get_output(act_i) for act_i in activations ]
            func_activations = [theano.function([inputs], [activations[idx]]) for idx in range(6) ]
            #make train True from command line argument and uncomment below if want to test activations pruning but dont want to train
            #train = False
        new_param_values, filter_sizes = compress.kernel_filter_pruning_functionality(
           filter_pruning_type, params_binary, param_values, filter_percentage_prune,
           network_type, validation_data, func_activations, batch_size)
        cnn, activations, train_fn, val_fn, activ = cifar_models.build_model()
        lasagne.layers.set_all_param_values(cnn, new_param_values)
    #train network with or without pruning
    print(train)
    if train == True:
        print('Training...')
        if prune == False:
            binary_net.train(
                    train_fn,val_fn,
                    cnn, filter_percentage_prune, filter_pruning_type, activ_output,
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

            binary_net.train_prune( train_fn,val_fn,
                                    cnn,
                                    batch_size,
                                    LR_start,LR_decay,
                                    num_epochs,
                                    train_set.X,train_set.y,
                                    valid_set.X,valid_set.y,
                                    test_set.X,test_set.y, H,
                                    binary, Masker1, Masker2,
                                    Masker3, Masker4, Masker5,
                                    Masker6, Masker7, Masker8,
                                    Masker9, shuffle_parts=shuffle_parts)
