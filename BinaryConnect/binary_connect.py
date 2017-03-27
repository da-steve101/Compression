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

import time

from collections import OrderedDict

import numpy as np

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1')
import theano
import theano.tensor as T
import common
import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# This function computes the gradient of the binary weights
def compute_grads(loss,network):

    layers = lasagne.layers.get_all_layers(network)
    grads = []

    for layer in layers:

        params = layer.get_params(binary=True)
        if params:
            # print(params[0].name)
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

# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default train function in Lasagne yet)
def train(train_fn,val_fn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            X_train,y_train,
            X_val,y_val,cnn_pruned, percentage_prune, pruning_type,
            X_test,y_test):

    # A function which shuffles a dataset
    def shuffle(X,y):

        shuffled_range = range(len(X))
        np.random.shuffle(shuffled_range)
        # print(shuffled_range[0:10])

        new_X = np.copy(X)
        new_y = np.copy(y)

        for i in range(len(X)):

            new_X[i] = X[shuffled_range[i]]
            new_y[i] = y[shuffled_range[i]]

        return new_X,new_y

    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X,y,LR):

        loss = 0
        batches = len(X)/batch_size

        for i in range(batches):
            loss += train_fn(X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size],LR)

        loss/=batches

        return loss

    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X,y):

        err = 0
        loss = 0
        batches = len(X)/batch_size

        for i in range(batches):
            new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            err += new_err
            loss += new_loss

        err = err / batches * 100
        loss /= batches

        return err, loss

    # shuffle the train set
    X_train,y_train = shuffle(X_train,y_train)
    best_val_err = 100
    best_epoch = 1
    LR = LR_start

    # We iterate over epochs:
    for epoch in range(num_epochs):

        start_time = time.time()

        train_loss = train_epoch(X_train,y_train,LR)
        X_train,y_train = shuffle(X_train,y_train)

        val_err, val_loss = val_epoch(X_val,y_val)

        # test if validation error went down
        if val_err <= best_val_err:

            best_val_err = val_err
            best_epoch = epoch+1

            test_err, test_loss = val_epoch(X_test,y_test)

            f = open('cnn_binarized_'+ str(str(pruning_percentage).replace(".","")) + '_' + str(pruning_type)+'.save', 'wb')
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
