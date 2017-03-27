
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

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

# Our own rounding function, that does not set the gradient to 0 like Theano's

round3_scalar = common.Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)

# This function computes the gradient of the binary weights
def compute_grads(loss,network, Masker1, Masker2, Masker3, Masker4, Masker5, Masker6, Masker7, Masker8, Masker9):

    layers = lasagne.layers.get_all_layers(network)
    grads = []
    Masks = [Masker1, Masker2, Masker3, Masker4, Masker5, Masker6, Masker7, Masker8, Masker9]
    i=0
    for layer in layers:

        params = layer.get_params(binary=True)
        if params:
            # print(params[0].name)
            grads.append(theano.grad(loss, wrt=layer.Wb)*Masks[i])
            #grads.append(theano.grad(loss, wrt=layer.Wb))
            i+=1
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
# (There is no default trainer function in Lasagne yet)
def train(train_fn,val_fn,
          model,
          batch_size,
          LR_start,LR_decay,
          num_epochs,
          X_train,y_train,
          X_val,y_val,
          X_test,y_test, H,
          binary, Masker1,
          Masker2, Masker3,
          Masker4, Masker5,
          Masker6, Masker7,
          Masker8, Masker9,
          shuffle_parts=1):

    # A function which shuffles a dataset
    def shuffle(X,y):

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
    def train_epoch(X,y,LR, Layer1_mask, Layer2_mask, Layer3_mask,
                    Layer4_mask, Layer5_mask, Layer6_mask, Layer7_mask,
                    Layer8_mask, Layer9_mask):

        loss = 0
        batches = len(X)/batch_size

        for i in range(batches):
            loss += train_fn(X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size],LR, Layer1_mask, Layer2_mask, Layer3_mask, Layer4_mask, Layer5_mask, Layer6_mask, Layer7_mask, Layer8_mask, Layer9_mask)#, HW)

        loss/=batches

        return loss

    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X,y, Layer1_mask, Layer2_mask, Layer3_mask, Layer4_mask, Layer5_mask, Layer6_mask, Layer7_mask, Layer8_mask, Layer9_mask):

        err = 0
        loss = 0
        batches = len(X)/batch_size

        for i in range(batches):
            new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size], Layer1_mask, Layer2_mask, Layer3_mask, Layer4_mask, Layer5_mask, Layer6_mask, Layer7_mask, Layer8_mask, Layer9_mask)
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
    layers_NZ = lasagne.layers.get_all_layers(cnn)
    print(layers_NZ)
    Nonzeros = ((layers_NZ[1].Wb)**2).sum() + ((layers_NZ[3].Wb)**2).sum() + ((layers_NZ[6].Wb)**2).sum() + ((layers_NZ[8].Wb)**2).sum() + ((layers_NZ[11].Wb)**2).sum() + ((layers_NZ[13].Wb)**2).sum() + ((layers_NZ[16].Wb)**2).sum() + ((layers_NZ[18].Wb)**2).sum() + ((layers_NZ[20].Wb)**2).sum()
    print(Nonzeros.eval({layers_NZ[1].Layer_mask : Masker1, layers_NZ[3].Layer_mask : Masker2, layers_NZ[6].Layer_mask : Masker3, layers_NZ[8].Layer_mask : Masker4, layers_NZ[11].Layer_mask : Masker5, layers_NZ[13].Layer_mask : Masker6, layers_NZ[16].Layer_mask : Masker7, layers_NZ[18].Layer_mask : Masker8, layers_NZ[20].Layer_mask : Masker9}))

    # We iterate over epochs:
    for epoch in range(num_epochs):

        start_time = time.time()

        print(Nonzeros.eval({layers_NZ[1].Layer_mask : Masker1, layers_NZ[3].Layer_mask : Masker2, layers_NZ[6].Layer_mask : Masker3, layers_NZ[8].Layer_mask : Masker4, layers_NZ[11].Layer_mask : Masker5, layers_NZ[13].Layer_mask : Masker6, layers_NZ[16].Layer_mask : Masker7, layers_NZ[18].Layer_mask : Masker8, layers_NZ[20].Layer_mask : Masker9}))
        train_loss = train_epoch(X_train,y_train,LR, Masker1, Masker2, Masker3, Masker4, Masker5, Masker6, Masker7, Masker8, Masker9)

        X_train,y_train = shuffle(X_train,y_train)

        val_err, val_loss = val_epoch(X_val,y_val, Masker1, Masker2, Masker3, Masker4, Masker5, Masker6, Masker7, Masker8, Masker9)

        # test if validation error went down
        if val_err <= best_val_err:

            best_val_err = val_err
            best_epoch = epoch+1

            test_err, test_loss = val_epoch(X_test,y_test, Masker1, Masker2, Masker3, Masker4, Masker5, Masker6, Masker7, Masker8, Masker9)

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
