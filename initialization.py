import numpy
import theano
from fractions import gcd

from blocks.initialization import NdarrayInitialization


class NormalizedInitialization(NdarrayInitialization):
    """Initialize parameters with Glorot method.

    Notes
    -----
    For details see
    Understanding the difficulty of training deep feedforward neural networks,
    Glorot, Bengio, 2010

    """
    def generate(self, rng, shape):
        # In the case of diagonal matrix, we initialize the diagonal
        # to zero. This may happen in LSTM for the weights from cell
        # to gates.
        if len(shape) == 1:
            m = numpy.zeros(shape=shape)
        else:
            input_size, output_size = shape
            high = numpy.sqrt(6) / numpy.sqrt(input_size + output_size)
            m = rng.uniform(-high, high, size=shape)
        return m.astype(theano.config.floatX)


class IdentityInitialization(NdarrayInitialization):
    """ Initialize parameters with I * c."""
    def __init__(self, c):
        self.c = c

    def generate(self, rng, shape):
        return self.c * numpy.eye(*shape, dtype=theano.config.floatX)




class OrthogonalInitialization(NdarrayInitialization):
    # Janos Kramar
    def generate(self, rng, shape):
        W = rng.normal(0.0, 1.0, shape)
        #factor = gcd(*W.shape)
        #assert factor in W.shape
        #for i in range(W.shape[0]/factor):
            #for j in range(W.shape[1]/factor):
                #W[factor*i:factor*(i+1),factor*j:factor*(j+1)], _, _ = numpy.linalg.svd(W[factor*i:factor*(i+1),factor*j:factor*(j+1)])
        #return W.astype(theano.config.floatX)
        u,s,v = numpy.linalg.svd(W,False)
        return u.astype(theano.config.floatX) if u.shape == W.shape else v.astype(theano.config.floatX)
    