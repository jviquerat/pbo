# Generic imports
import math
import numpy as np

# Custom imports
from base_env import base_env

###############################################
### Environment for himmelblau function
class himmelblau(base_env):

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'rosenbrock_2d'
        self.act_size = 2
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        self.x_min    = np.array([-4.0,-4.0])
        self.x_max    = np.array([ 4.0, 4.0])
        self.x_0      = np.array([ 0.0, 0.0])

    ### Actual function
    def function(self, x):

        # Compute function value in x
        val = (x[0]**2 + x[1] - 11.0)**2 + (x[0] + x[1]**2 - 7.0)**2

        return -val
