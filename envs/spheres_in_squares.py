# Generic imports
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Custom imports
from base_env import base_env

###############################################
### Environment for the packing of spheres in squares
class spheres_in_squares(base_env):

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.n_spheres  = 6
        self.max_side   = 7.0
        self.radius     = 1.0

        self.name      = 'spheres_in_squares'
        self.act_size  = 2*self.n_spheres
        self.obs_size  = self.act_size
        self.obs       = np.zeros(self.obs_size)
        self.x_min     = np.zeros(self.act_size) + self.radius
        self.x_max     = np.ones(self.act_size)*self.max_side - self.radius
        self.bst_rwd   = -1.0e8

        # Compute x_0
        self.x_0       = 0.5*(self.x_min + self.x_max)

        # Make path
        self.path      = path
        os.makedirs(self.path, exist_ok=True)

    ### Take one step
    def step(self, actions, ep):

        # Convert actions
        x = self.convert_actions(actions)

        # Retrieve coordinates
        j      = 0
        coords = np.zeros((self.n_spheres,2))
        for i in range(self.n_spheres):
            coords[i,0] = x[j]
            coords[i,1] = x[j+1]
            j          += 2

        # Compute reward
        ri, rs, xm, ym, side = self.function(coords)
        rwd = ri + rs

        # Plot
        if (rwd > self.bst_rwd):
            self.bst_rwd = rwd
            self.plot(coords, xm, ym, side, ri, rs, ep)

        return rwd, x

    ### Plot
    def plot(self, c, x, y, s, ri, rs, ep):

        # Axis
        fig, ax = plt.subplots()
        ax.set_xlim([0, self.max_side])
        ax.set_ylim([0, self.max_side])
        plt.gca().set_aspect('equal', adjustable='box')

        # Add circles
        for i in range(self.n_spheres):
            circle = plt.Circle((c[i,0], c[i,1]), self.radius,
                                fc='gray', ec='black')
            ax.add_patch(circle)

        # Add square
        rectangle = plt.Rectangle((x,y), s, s, fc='none', ec='black', lw=2)
        plt.gca().add_patch(rectangle)

        # Add title
        fi = f"{ri:.5f}"
        fs = f"{rs:.5f}"
        plt.title("ep "+str(ep)+", ri="+fi+", rs="+fs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        # Save figure and close
        fig.savefig(self.path+'/'+str(ep)+'.png',
                    bbox_inches='tight')
        plt.close(plt.gcf())
        plt.cla()

    ### Actual function
    def function(self, coords):

        # Set rewards
        ri = 0.0 # intersection
        rs = 0.0 # side

        # Check intersections
        dsum = 0.0
        for i in range(self.n_spheres):
            for j in range(self.n_spheres):
                if (i == j): continue

                x0    = coords[i,0]
                y0    = coords[i,1]
                x1    = coords[j,0]
                y1    = coords[j,1]
                d     = math.sqrt((x1-x0)**2 + (y1-y0)**2)

                # Optimal is d=2*radius
                d -= 2.0*self.radius
                if (d < 0.0): ri -= 5.0*abs(d)

        # Compute min square side
        x_min = 1.0e8
        x_max =-1.0e8
        y_min = 1.0e8
        y_max =-1.0e8

        for i in range(self.n_spheres):
            x0 = coords[i,0]
            y0 = coords[i,1]
            x_min = min(x0, x_min)
            x_max = max(x0, x_max)
            y_min = min(y0, y_min)
            y_max = max(y0, y_max)

        x_min -= self.radius
        y_min -= self.radius
        x_max += self.radius
        y_max += self.radius
        side   = max(x_max-x_min, y_max-y_min)
        rs    -= side

        return ri, rs, x_min, y_min, side
