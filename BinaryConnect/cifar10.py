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
    def load_model(filename, model):
        f = open(str(filename), 'rb')
        loadedobj = cPickle.load(f)
        lasagne.layers.set_all_param_values(model, loadedobj)
        f.close()
    
        return model

    #convert all real valued weights into ternary values
    def ternarize_cnn(params, conv_thresh, fc_thresh):
        #for cnn w/out BA, weight param indexes for conv and fc are 0,6,12,...,48
        conv = [params[0],params[1], params[2], params[3], params[4], params[5]]
        fc = [params[6], params[7], params[8]]
        count = 0

        for i in conv:
            i = np.clip(i,-1., 1.)
            i = np.select([i<-conv_thresh, i>conv_thresh, (-conv_thresh <= i) & (i <= conv_thresh)], [-1, 1 , 0])
            i = i.astype(np.float32)
            conv[count] = i
            count += 1
        count = 0
        for j in fc:
            j = np.clip(j,-1., 1.)
            j = np.select([j<-fc_thresh, j>fc_thresh, (-fc_thresh <= j) & (j <= fc_thresh)], [-1, 1 , 0])
            j = j.astype(np.float32)
            fc[count] = j
            count +=1

        params_ternarized = conv + fc

        return params_ternarized

    def binarize_cnn(params):
        conv = [params[0],params[1], params[2], params[3], params[4], params[5]]
        fc = [params[6], params[7], params[8]]
        count = 0

        for i in conv:
            i = np.clip((i+1.)/2.,0.,1.)
            i = np.around(i)
            i = np.select([i<1., i>0.], [-1., 1.])
            i = i.astype(np.float32)
            conv[count] = i
            count += 1
        count = 0

        for j in fc:
            j = np.clip((j+1.)/2.,0.,1.)
            j = np.around(j)
            j = np.select([j<1., j>0.], [-1., 1.])
            j = j.astype(np.float32)
            fc[count] = j
            count +=1

        params_ternarized = conv + fc

        return params_ternarized

    #count all preceding zeros for convolution layer
    def count_zeros_conv(kernel, index1,index2,index3, index4):
        #Enter filter kernel and its indexes from '.shape' = (index1,index2, index3,index4)
        counter_pos = []
        counter_neg = []
        counter = []
        count = 0
        for k in range(index4):
            for l in range(index3):
                for i in range(index2):
                    for j in range(index1):
                        if (kernel[j][i][l][k] != 0.):
                            counter.append(count)
                            if (kernel[j][i][l][k] == 1.):
                                counter_pos.append(count)
                                count = 0
                            elif (kernel[j][i][l][k] == -1.):
                                counter_neg.append(count)
                                count = 0
                            else:  
                                print("error")
                        else:
                            count += 1

        data1 = Counter(counter_pos).most_common()
        data2 = Counter(counter_neg).most_common() 
        data3 = Counter(counter)
        print(data3)
        print(data1)
        print(data2)

    #calculate absolute sum of all kernels in each layer and return a list of magnitudes for each layer and its normalized version
    def filter_magnitudes(layers):
        conv = [layers[0],layers[1], layers[2], layers[3], layers[4], layers[5]]
        magnitude = []
        magnitude_layers = []
        mag = 0

        #calculate the absolute magnitudes of each kernel
        for q in conv:
            for k in range(q.shape[0]):
                for l in range(q.shape[1]):
                    for i in range(q.shape[2]):
                        for j in range(q.shape[3]):
                            mag += abs(q[k][l][i][j])
                magnitude.append(mag)
                mag = 0
            magnitude_layers.append(magnitude)
            magnitude = []

        #calculate the normalized values and sort in descending order
        abs_sum = []
        tmp=[]
        for r in magnitude_layers:
            for y in range(len(r)):
                r[y] = float(r[y])
                s = r[y]/float(max(r))
                tmp.append(s)
            tmp = sorted(tmp, reverse=True)
            abs_sum.append(tmp)
            tmp = []

        return magnitude_layers, abs_sum

    #calculate the sum of quantized magnitudes of each kernel and then use the absolute value to see the kernel sums closest to zero.
    #outputs the tuple: (absolute magnitudes, index filter number) and the normalized values in descending order
    def filter_quantized_sum(layers):
        conv = [layers[0],layers[1], layers[2], layers[3], layers[4], layers[5]]
        magnitude = []
        magnitude_layers = []
        maximums = []
        mag = 0

        #calculate the sum of quantized magnitudes of each kernel and then use the absolute value to see the kerneal sums closes to zero
        for q in conv:
            for k in range(q.shape[0]):
                for l in range(q.shape[1]):
                    for i in range(q.shape[2]):
                        for j in range(q.shape[3]):
                            mag += q[k][l][i][j]
                magnitude.append(abs(mag))
                mag = 0
            maximums.append(max(magnitude))
            magnitude_layers.append(magnitude)
            magnitude = []

        #sort magnitudes in ascending order and make a tuple by adding their index number (value,index) 
        # so that we know we cut off the first elements and then we have their indexes to know which parameters to remove from the cnn
        for i in range(len(magnitude_layers)):
            for j in range(len(magnitude_layers[i])):
                magnitude_layers[i][j] = (magnitude_layers[i][j], j)

        magnitude_layers_sorted = []
        for i in magnitude_layers:
            tmp = sorted(i, key=lambda tup: tup[0])
            magnitude_layers_sorted.append(tmp)

        #calculate the normalized values and sort in descending order (for the plots) where max value is the 
        # maximum absolute value (i.e value furtherest from zero)
        normalized_sums = []
        tmp=[]
        for r in range(len(magnitude_layers)):
            for y in range(len(magnitude_layers[r])):
                s = float(magnitude_layers[r][y][0])/float(maximums[r])
                tmp.append(s)
            tmp = sorted(tmp, reverse=True)
            normalized_sums.append(tmp)
            tmp = []

        return magnitude_layers_sorted, normalized_sums

    #randomly generate which filters will be pruned and output new list of params with dimensiond given by 'filter_sizes' output
    def random_pruning(params, param_values,saved_filter_percentage):

        filters = []
        j=0
        for i in params:
            if (str(i) == 'W'):
                filters.append(param_values[j].shape[0])
            j+=1
        #remove fc layer sizes
        filters.pop()
        filters.pop()
        filters.pop()

        random = []
        new_filter_sizes = []

        for i in range(len(filters)):
            random.append(np.random.binomial(1,saved_filter_percentage, size=(1,filters[i]))[0])
            #print(np.sum(random[i]))
        for p in xrange(0,(len(random)*6), 6):
            j=0
            for i in range(len(random[(int(float(p)/float(6)))])):
                e = 16*i
                if (random[(int(float(p)/float(6)))][i]==0):
                    param_values[p] = np.delete(param_values[p], i-j,0)
                    if (p<(len(filters)*5)):
                        param_values[p+1] = np.delete(param_values[p+1], i-j,0)
                        param_values[p+2] = np.delete(param_values[p+2], i-j,1)
                        param_values[p+3] = np.delete(param_values[p+3], i-j,1)
                        param_values[p+4] = np.delete(param_values[p+4], i-j,1)
                        param_values[p+5] = np.delete(param_values[p+5], i-j,1)
                        param_values[p+6] = np.delete(param_values[p+6], i-j,1)
                    elif (p == 30):
                        param_values[p+1] = np.delete(param_values[p+1], i-j,0)
                        param_values[p+2] = np.delete(param_values[p+2], i-j,1)
                        param_values[p+3] = np.delete(param_values[p+3], i-j,1)
                        param_values[p+4] = np.delete(param_values[p+4], i-j,1)
                        param_values[p+5] = np.delete(param_values[p+5], i-j,1)
                        param_values[p+6] = np.delete(param_values[p+6], [(e-(16*j)),(e-(16*j))+1,(e-(16*j))+2,(e-(16*j))+3,(e-(16*j))+4,(e-(16*j))+5,(e-(16*j))+6,(e-(16*j))+7,(e-(16*j))+8,(e-(16*j))+9,(e-(16*j))+10,(e-(16*j))+11,(e-(16*j))+12,(e-(16*j))+13,(e-(16*j))+14,(e-(16*j))+15] ,0)
                    j+=1

            new_filter_sizes.append(param_values[p].shape[0])

        return param_values, new_filter_sizes

    cnn = build_model(128,128,256,256,512,512)

    print('Training...')
    
    #tester = lasagne.layers.get_output(cnn1, deterministic=True)

    # tester_fn = theano.function([input], tester)

    # ones = np.ones((1,3,32,32))


    # cnn = load_model('../../../../BinaryConnect/cnn_binarized.save', cnn)
    cnn = load_model('/home/jfar0131/job3/BinaryConnect/cnn_binarized.save', cnn)

    params_binary = lasagne.layers.get_all_param_values(cnn, binary=True)
    params = lasagne.layers.get_all_params(cnn)
    param_values = lasagne.layers.get_all_param_values(cnn)

    p = binarize_cnn(params_binary)

    #mags1, normalized = filter_magnitudes(params)
    mags1, normalized = filter_quantized_sum(p)
    new_param_values, filter_sizes = random_pruning(params, param_values,0.65)

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




    # Change to Xilinx computer and use github
    # Use counter function on sydney uni computer but change for loop to that you can count zeros in each filter
    # Add this function using the same process
    # After getting these statistics, try a mild pruning strategy of the filters and then retrain the network 

    #find magnitude of each filter for each layer and sort it  



