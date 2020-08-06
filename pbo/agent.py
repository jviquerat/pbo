# Generic imports
import os
import warnings

# Import tensorflow and filter warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow                    as     tf
import tensorflow.keras              as     tk
import tensorflow_probability        as     tfp
import tensorflow_addons             as     tfa
from   tensorflow.keras              import Model
from   tensorflow.keras.layers       import Dense
from   tensorflow.keras.initializers import Orthogonal

# Define alias
tf.keras.backend.set_floatx('float64')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
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
                                  activation = 'tanh',
                                  use_bias=False,
                                  dtype='float64'))

        # Define last layer
        self.net.append(Dense(dim,
                              kernel_initializer=Orthogonal(gain=0.01),
                              activation = last,
                              use_bias=False,
                              dtype='float64'))

        # Define optimizer
        self.opt = tk.optimizers.Adam(lr       = lr,
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
