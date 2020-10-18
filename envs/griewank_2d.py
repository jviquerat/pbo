# Generic imports
import numpy as     np
from   math  import cos, sqrt, pi

# Custom imports
from envs.base_env import base_env

###############################################
### Environment for griewank_2d function
class griewank_2d(base_env):

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'griewank_2d'
        self.act_size = 2
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        self.x_min    = np.array([-10.0,-10.0])
        self.x_max    = np.array([ 10.0, 10.0])
        self.x_0      = np.array([  5.0,  5.0])

    ### Actual function
    def function(self, x):

        return -(1.0 + (x[0]**2+x[1]**2)/4000.0 - cos(x[0])*cos(x[1]/sqrt(2.0)))
