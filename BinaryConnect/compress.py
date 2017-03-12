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

def binarize_cnn(param_values_binary):
    conv = [param_values_binary[0],param_values_binary[1], param_values_binary[2], param_values_binary[3], param_values_binary[4], param_values_binary[5]]
    fc = [param_values_binary[6], param_values_binary[7], param_values_binary[8]]
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
def filter_magnitudes(param_values_binary):
    conv = [param_values_binary[0],param_values_binary[1], param_values_binary[2], param_values_binary[3], param_values_binary[4], param_values_binary[5]]
    magnitude = []
    magnitude_layers = []
    maximums = []
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

    #calculate the normalized values and sort in descending order
    normalized_sum = []
    tmp=[]
    for r in range(len(magnitude_layers)):
        for y in range(len(magnitude_layers[r])):
            s = float(magnitude_layers[r][y][0])/float(maximums[r])
            tmp.append(s)
        tmp = sorted(tmp, reverse=True)
        normalized_sum.append(tmp)
        tmp = []

    return magnitude_layers, magnitude_layers_sorted, normalized_sum

#After obtaining a binary matrix to determine which filters to prune, we use this to prune the corresponding input/output filters for the whole network
def restructure_param_values(random, param_values, filters, network_type):
	#for binarynet p+2 --> p+5 = 0, binaryconnect p+2 --> p+5 = 1.
    if network_type == 'binarynet':
        ax = 0
    elif network_type == 'binaryconnect':
        ax = 1

    for p in xrange(0,(len(random)*6), 6):
        j=0
        for i in range(len(random[(int(float(p)/float(6)))])):
            e = 16*i
            if (random[(int(float(p)/float(6)))][i]==0):
                param_values[p] = np.delete(param_values[p], i-j,0)
                if (p<(len(filters)*5)):
                	param_values[p+1] = np.delete(param_values[p+1], i-j,0)
                	param_values[p+2] = np.delete(param_values[p+2], i-j,ax)
                	param_values[p+3] = np.delete(param_values[p+3], i-j,ax)
                	param_values[p+4] = np.delete(param_values[p+4], i-j,ax)
                	param_values[p+5] = np.delete(param_values[p+5], i-j,ax)
                	param_values[p+6] = np.delete(param_values[p+6], i-j,1)
                elif (p == 30):
                    param_values[p+1] = np.delete(param_values[p+1], i-j,0)
                    param_values[p+2] = np.delete(param_values[p+2], i-j,ax)
                    param_values[p+3] = np.delete(param_values[p+3], i-j,ax)
                    param_values[p+4] = np.delete(param_values[p+4], i-j,ax)
                    param_values[p+5] = np.delete(param_values[p+5], i-j,ax)
                    param_values[p+6] = np.delete(param_values[p+6], [(e-(16*j)),(e-(16*j))+1,(e-(16*j))+2,(e-(16*j))+3,(e-(16*j))+4,(e-(16*j))+5,(e-(16*j))+6,(e-(16*j))+7,(e-(16*j))+8,(e-(16*j))+9,(e-(16*j))+10,(e-(16*j))+11,(e-(16*j))+12,(e-(16*j))+13,(e-(16*j))+14,(e-(16*j))+15] ,0)
                j+=1
        print(param_values[p].shape)
        print(param_values[p+1].shape)
        print(param_values[p+2].shape)
        print(param_values[p+3].shape)
        print(param_values[p+4].shape)
        print(param_values[p+5].shape)
    # print(param_values[36].shape)
    # print(param_values[42].shape)
    # print(param_values[48].shape)
    return param_values

#calculate the sum of quantized magnitudes of each kernel and then use the absolute value to see the kernel sums closest to zero.
#input manually binarized parameters
#outputs magnitude tuples: (absolute magnitudes, index filter number) in order and also in ascending order and also the normalized values in descending order
def filter_quantized_sum(quantized_params):
    conv = [quantized_params[0],quantized_params[1], quantized_params[2], quantized_params[3], quantized_params[4], quantized_params[5]]
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

    magnitude_layers_ascending = []
    for i in magnitude_layers:
        tmp = sorted(i, key=lambda tup: tup[0])
        magnitude_layers_ascending.append(tmp)

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

    return magnitude_layers, magnitude_layers_ascending, normalized_sums

#randomly generate which filters will be pruned and output new list of params with dimensiond given by 'filter_sizes' output
def random_pruning(param_values_binary, param_values,saved_filter_percentage, network_type):

    filters = []
    for i in param_values_binary:
        filters.append(i.shape[0])
    #remove fc layer sizes
    filters.pop()
    filters.pop()
    filters.pop()

    random = []
    new_filter_sizes = []

    for i in range(len(filters)):
        random.append(np.random.binomial(1,saved_filter_percentage, size=(1,filters[i]))[0])
        new_filter_sizes.append(np.sum(random[i]))
        print(new_filter_sizes[i])

    print(filters)
    print(len(random))
    print(random[0])
    print(new_filter_sizes)
    for i in param_values_binary:
    	print(i.shape)
    for i in param_values:
    	print(i.shape)    

    param_values = restructure_param_values(random, param_values, filters, network_type)

    return param_values, new_filter_sizes

#prune filters which have the smallest absolute sum of the real-valued weights
def real_weights_pruning(param_values_binary, param_values,saved_filter_percentage, network_type):
    number_of_conv_layers = 3
    filters = []
    new_filters = []

    #get two lists which define the number of parameters for each layer of the original and pruned networks 
    for i in param_values_binary:
        filters.append(i.shape[0])
        new_filters.append(int(np.around(saved_filter_percentage*i.shape[0])))
    #remove fc layer sizes
    for i in range(number_of_conv_layers):
        filters.pop()
        new_filters.pop()

    mags1, mags1_ascending, normalized = filter_magnitudes(param_values_binary)
    
    keep_params = []
    random = []

    #create matrix of ones and zeros which indicates whether to keep a particular filter
    for i in range(len(mags1_ascending)):
        random.append(np.zeros(filters[i]))
        keep_params.append(mags1_ascending[i][(filters[i] - new_filters[i]):])
        for j in keep_params[i]:
            random[i][j[1]] = 1

    param_values = restructure_param_values(random, param_values, filters, network_type)
        
    return param_values, new_filters

#prune filters which have the smallest sum of quantized weights
def quantized_weights_pruning(param_values_binary, param_values,saved_filter_percentage, network_type):
    number_of_conv_layers = 3
    filters = []
    new_filters = []

    #get two lists which define the number of parameters for each layer of the original and pruned networks 
    for i in param_values_binary:
        filters.append(i.shape[0])
        new_filters.append(int(np.around(saved_filter_percentage*i.shape[0])))
    #remove fc layer sizes
    for i in range(number_of_conv_layers):
        filters.pop()
        new_filters.pop()

    binarized_params = binarize_cnn(param_values_binary)

    mags1, mags1_ascending, normalized = filter_quantized_sum(binarized_params)
    print(len(mags1))
    

    keep_params = []
    random = []

    #create matrix of ones and zeros which indicates whether to keep a particular filter
    for i in range(len(mags1_ascending)):
        random.append(np.zeros(filters[i]))
        keep_params.append(mags1_ascending[i][(filters[i] - new_filters[i]):])
        for j in keep_params[i]:
            random[i][j[1]] = 1

    print(random[0])
    print(len(keep_params[0]))

    param_values = restructure_param_values(random, param_values, filters, network_type)
    
    return param_values, new_filters
