# Custom imports
from envs.lorenz import *

###############################################
### Environment for Lorenz oscillator
class lorenz_oscillator(lorenz):

    # Create object
    def __init__(self, path):
        super().__init__(path)

        # Fill structure
        self.name = 'lorenz_oscillator'

    # Compute reward
    def get_rwd(self):

        rwd = 0.0
        for i in range(len(self.lst_x)-1):
            if (self.lst_x[i]*self.lst_x[i+1] < 0.0): rwd += 1

        return rwd
