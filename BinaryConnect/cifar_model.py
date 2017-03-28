
import theano
import theano.tensor as T
import common
import lasagne
import binary_net

from collections import OrderedDict

class Cifar_Model:
    def __init__( self, inputs, target, filter_sizes ):
        self.inputs = inputs
        self.target = target
        self.filter_sizes = filter_sizes
        self.layer_masks = []
        self.activation_fn = None
        
    @staticmethod
    def getNonLinearLayer( cnn, activation_fn ):
        if activation_fn is not None:
            return lasagne.layers.NonlinearityLayer(
                cnn,
                nonlinearity=activation_fn)
        return lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=lasagne.nonlinearities.rectify)

    def getMask( self,  idx ):
        if idx < len( self.layer_masks ):
            return self.layer_masks[idx]
        return None
    
    def build_model( self ):

        activations = []

        cnn = lasagne.layers.InputLayer(
            shape=(None, 3, 32, 32),
            input_var = self.inputs )

        # 128C3-128C3-P2
        cnn = common.Conv2DLayer(
            cnn,
            binary = self.binary,
            stochastic = self.stochastic,
            H = self.H,
            W_LR_scale = self.W_LR_scale,
            num_filters = self.filter_sizes[0],
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
        cnn.layer_mask = self.getMask( 0 )
        
        cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon = self.epsilon,
            alpha = self.alpha)

        activations += [cnn]
        
        cnn = self.getNonLinearLayer( cnn, self.activation_fn )

        activ = cnn

        cnn = common.Conv2DLayer(
            cnn,
            binary = self.binary,
            stochastic = self.stochastic,
            H = self.H,
            W_LR_scale = self.W_LR_scale,
            num_filters = self.filter_sizes[1],
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
        cnn.layer_mask = self.getMask( 1 )
        
        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

        cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon = self.epsilon,
            alpha = self.alpha)

        activations += [cnn]
        
        cnn = self.getNonLinearLayer( cnn, self.activation_fn )

        # 256C3-256C3-P2
        cnn = common.Conv2DLayer(
            cnn,
            binary = self.binary,
            stochastic = self.stochastic,
            H = self.H,
            W_LR_scale = self.W_LR_scale,
            num_filters = self.filter_sizes[2],
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
        cnn.layer_mask = self.getMask( 2 )
        
        cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon = self.epsilon,
            alpha = self.alpha)

        activations += [cnn]

        cnn = self.getNonLinearLayer( cnn, self.activation_fn )

        cnn = common.Conv2DLayer(
            cnn,
            binary = self.binary,
            stochastic = self.stochastic,
            H = self.H,
            W_LR_scale = self.W_LR_scale,
            num_filters = self.filter_sizes[3],
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
        cnn.layer_mask = self.getMask( 3 )
        
        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
        
        cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon = self.epsilon,
            alpha = self.alpha)

        activations += [cnn]

        cnn = self.getNonLinearLayer( cnn, self.activation_fn )

        # 512C3-512C3-P2
        cnn = common.Conv2DLayer(
            cnn,
            binary = self.binary,
            stochastic = self.stochastic,
            H = self.H,
            W_LR_scale = self.W_LR_scale,
            num_filters = self.filter_sizes[4],
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
        cnn.layer_mask = self.getMask( 4 )

        cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon = self.epsilon,
            alpha = self.alpha)

        activations += [cnn]

        cnn = self.getNonLinearLayer( cnn, self.activation_fn )

        cnn = common.Conv2DLayer(
            cnn,
            binary = self.binary,
            stochastic = self.stochastic,
            H = self.H,
            W_LR_scale = self.W_LR_scale,
            num_filters = self.filter_sizes[5],
            filter_size=(3, 3),
            pad=1,
            nonlinearity=lasagne.nonlinearities.identity)
        cnn.layer_mask = self.getMask( 5 )
        
        cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

        cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon = self.epsilon,
            alpha = self.alpha)

        activations += [cnn]

        cnn = self.getNonLinearLayer( cnn, self.activation_fn )
        lasagne.nonlinearities.rectify

        # print(cnn.output_shape)

        # 1024FP-1024FP-10FP
        cnn = common.DenseLayer(
            cnn,
            binary = self.binary,
            stochastic = self.stochastic,
            H = self.H,
            W_LR_scale = self.W_LR_scale,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=1024)
        cnn.layer_mask = self.getMask( 6 )
        
        cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon = self.epsilon,
            alpha = self.alpha)

        cnn = self.getNonLinearLayer( cnn, self.activation_fn )
        
        cnn = common.DenseLayer(
            cnn,
            binary = self.binary,
            stochastic = self.stochastic,
            H = self.H,
            W_LR_scale = self.W_LR_scale,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=1024)
        cnn.layer_mask = self.getMask( 7 ),        

        cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon = self.epsilon,
            alpha = self.alpha)
        
        cnn = self.getNonLinearLayer( cnn, self.activation_fn )
    
        cnn = common.DenseLayer(
            cnn,
            binary = self.binary,
            stochastic = self.stochastic,
            H = self.H,
            W_LR_scale = self.W_LR_scale,
            nonlinearity=lasagne.nonlinearities.identity,
            num_units=10)
        cnn.layer_mask = self.getMask( 8 )

        cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon = self.epsilon,
            alpha = self.alpha)

        if self.activation_fn is None:
            return cnn

        train_output = lasagne.layers.get_output(cnn, deterministic=False)
    
        HW = 0
        layers_NZ = lasagne.layers.get_all_layers(cnn)
        masked_layers = [ l.Wb for l in layers_NZ if hasattr( l, 'layer_mask' ) and l.layer_mask != None ]
        
        if len(masked_layers) != 0:
            L2_Regularization = sum([ lasagne.regularization.l2(lWb) for lWb in masked_layers ])
            HW = 0.00000000412*L2_Regularization
    
        # squared hinge loss
        loss = T.mean(T.sqr(T.maximum(0.,1.-self.target*train_output)))

        if self.binary:
            # W updates
            W = lasagne.layers.get_all_params(cnn, binary=True)
            W_grads = binary_net.compute_grads(loss,cnn)
            updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=self.LR)
            updates = binary_net.clipping_scaling(updates,cnn)

            # other parameters updates
            params = lasagne.layers.get_all_params(cnn, trainable=True, binary=False)
            updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=self.LR).items())
        else:
            params = lasagne.layers.get_all_params(cnn, trainable=True)
            updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=self.LR)

        test_output = lasagne.layers.get_output(cnn, deterministic=True)
        test_loss = T.mean(T.sqr(T.maximum(0.,1.-self.target*test_output))) + HW
        test_loss = T.mean(T.sqr(T.maximum(0.,1.-self.target*test_output)))
        test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(self.target, axis=1)),dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving the updates dictionary)
        # and returning the corresponding training loss:
        train_fn = theano.function([self.inputs, self.target, self.LR ] + self.layer_masks, loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([self.inputs, self.target ] + self.layer_masks, [test_loss, test_err])

        return cnn, activations, train_fn, val_fn, activ

    def build_model_prune_and_L2( self, activation_fn ):
        # have this assertion as the fact that activation is None used to choose net output
        assert self.activation_fn is not None, "Activation function cannot be none for L2 and prune"
        self.activation_fn = activation_fn
        self.numFilters = [ 128, 128, 256, 256, 512, 512 ]
        conv_masks = [ T.ftensor4('Layer' + str(idx + 1) + '_mask') for idx in range(6) ]
        dense_masks = [ T.fmatrix('Layer' + str(idx + 7) + '_mask') for idx in range(3) ]
        self.layer_masks = conv_masks + dense_masks
        return build_model( )

