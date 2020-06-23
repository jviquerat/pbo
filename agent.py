# Generic imports
import os
import warnings

# Import tensorflow and filter warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow                    as     tf
import tensorflow.keras              as     tk
import tensorflow_probability        as     tfp
from   tensorflow.keras              import Model
from   tensorflow.keras.layers       import Dense
from   tensorflow.keras.initializers import Orthogonal

# Define alias
tfd = tfp.distributions

###############################################
### Neural network for mu and sg prediction
class nn(Model):
    def __init__(self, arch, dim, last, grd_clip, lr):
        super(nn, self).__init__()

        # Initialize network as empty list
        self.net = []

        # Define hidden layers
        for layer in range(len(arch)):
            self.net.append(Dense(arch[layer],
                                  kernel_initializer=Orthogonal(gain=1.0),
                                  activation = 'tanh'))

        # Define last layer
        self.net.append(Dense(dim,
                              kernel_initializer=Orthogonal(gain=0.01),
                              activation = last))

        # Define optimizer
        self.opt = tf.optimizers.Adam(lr       = lr,
                                      clipnorm = grd_clip)

    # Network forward pass
    @tf.function
    def call(self, var):

        # Copy input
        x = var

        # Compute output
        for layer in range(len(self.net)):
            x = self.net[layer](x)

        return x

###############################################
### Neural network specific for full cma
class nn_cma(Model):
    def __init__(self, arch, dim, grd_clip, lr):
        super(nn_cma, self).__init__()

        # Initialize network as empty list
        self.net = []

        # Define hidden layers
        for layer in range(len(arch)):
            self.net.append(Dense(arch[layer],
                                  kernel_initializer=Orthogonal(gain=1.0),
                                  activation = 'tanh'))

        # Diagonal terms, always positive
        self.net.append(Dense(dim,
                              kernel_initializer=Orthogonal(gain=0.01),
                              activation = 'softplus'))

        # Extra-diagonal terms, possibly negative
        self.net.append(Dense(int(dim*(dim-1)/2),
                              kernel_initializer=Orthogonal(gain=0.01),
                              activation = 'linear'))

        # Define optimizer
        self.opt = tf.optimizers.Adam(lr       = lr,
                                      clipnorm = grd_clip)

    # Network forward pass
    @tf.function
    def call(self, var):

        # Copy input
        x = var

        # Compute output
        depth = len(self.net)
        for layer in range(depth-2):
            x = self.net[layer](x)

        y = self.net[depth-2](x)
        z = self.net[depth-1](x)
        x = tf.concat([y,z],axis=1)

        return x
