# Generic imports
import numpy as     np
from   math  import cos, sqrt, pi

###############################################
### Environment for branin function
class branin():

    ### Create object
    def __init__(self):

        # Fill structure
        self.name     = 'branin'
        self.n_params = 2
        self.x_min    =-5.0
        self.x_max    = 10.0
        self.y_min    = 0.0
        self.y_max    = 15.0
        self.obs      = np.ones(1)

    ### Actual function
    def function(self, x, y):

        a = 1.0
        b = 5.1/(4.0*pi**2)
        c = 5.0/pi
        r = 6.0
        s = 10.0
        t = 1.0/(8.0*pi)

        return -(a*(y - b*x**2 + c*x - r)**2 + s*(1.0 - t)*cos(x) + s) + 0.397887

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
        conv_actions[0] = x_scale*actions[0] + 2.0
        conv_actions[1] = y_scale*actions[1] + 8.0

        return conv_actions

    ### Take one step
    def step(self, actions):

        # Take action and compute reward
        conv_actions = self.convert_actions(actions)
        x            = conv_actions[0]
        y            = conv_actions[1]
        reward       = self.function(x,y)

        return reward, conv_actions
