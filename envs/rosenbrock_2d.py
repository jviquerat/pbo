# Generic imports
import math
import numpy as np

# Custom imports
from base_env import base_env

###############################################
### Environment for rosenbrock_2d function
class rosenbrock_2d(base_env):

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'rosenbrock_2d'
        self.act_size = 2
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        self.x_min    = np.array([-2.0,-2.0])
        self.x_max    = np.array([ 2.0, 2.0])
        self.x_0      = np.array([ 0.0,-1.0])

    ### Actual function
    def function(self, x):

        # Compute function value in x
        val = 0.0
        for i in range(len(x)-1):
            val += 100.0*(x[i+1]-x[i]**2)**2 + (1.0-x[i])**2

        return -val
