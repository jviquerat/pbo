# Generic imports
import numpy as     np
from   math  import cos, sqrt, pi

###############################################
### Environment for griewank function
class griewank():

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'griewank'
        self.n_params = 2
        self.x_min    =-10.0
        self.x_max    = 10.0
        self.y_min    =-10.0
        self.y_max    = 10.0
        self.obs      = np.ones(1)

    ### Actual function
    def function(self, x, y):

        return -(1.0 + (x**2+y**2)/4000.0 - cos(x)*cos(y/sqrt(2.0)))

    ### Provide observation
    def observe(self):

        # Always return the same observation
        return self.obs

    ### Convert actions
    def convert_actions(self, actions):

        # Convert actions
        conv_actions    = self.n_params*[None]
        x_scale         = 0.5*(self.x_max - self.x_min)
        y_scale         = 0.5*(self.y_max - self.y_min)
        conv_actions[0] = x_scale*actions[0] + pi
        conv_actions[1] = y_scale*actions[1]

        return conv_actions

    ### Take one step
    def step(self, actions, ep):

        # Take action and compute reward
        conv_actions = self.convert_actions(actions)
        x            = conv_actions[0]
        y            = conv_actions[1]
        reward       = self.function(x,y)

        return reward, conv_actions
