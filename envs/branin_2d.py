# Generic imports
import numpy as     np
from   math  import cos, sqrt, pi

# Custom imports
from envs.base_env import base_env

###############################################
### Environment for branin_2d function
class branin_2d(base_env):

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'branin'
        self.act_size = 2
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        self.x_min    = np.array([ 0.0, 0.0])
        self.x_max    = np.array([15.0,15.0])
        self.x_0      = np.array([ 7.5, 7.5])

    ### Actual function
    def function(self, x):

        # Compute function value in x
        a = 1.0
        b = 5.1/(4.0*pi**2)
        c = 5.0/pi
        r = 6.0
        s = 10.0
        t = 1.0/(8.0*pi)

        return -(a*(x[1]-b*x[0]**2+c*x[0]-r)**2 + s*(1.0-t)*cos(x[0]) + s) + 0.397887
