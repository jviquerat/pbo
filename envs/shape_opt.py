# Generic imports
import numpy as np

# Custom imports
from utils.shapes  import *
from utils.solver  import *
from envs.base_env import *


###############################################
### Environment for shape optimization
class shape_opt():

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'shape_env'
        self.n_pts    = 4
        self.n_mv_pts = 1
        self.vol_pt   = 3
        self.n_params = self.vol_pt*self.n_mv_pts
        self.min_rad  = 0.2
        self.max_rad  = 1.0
        self.min_edg  = 0.0
        self.max_edg  = 1.0
        self.mv_pts   = [0]
        self.path     = path

        self.obs      = np.ones(1)
        self.shape    = generate_shape(self.n_pts, [0.0,0.0], 'cylinder',
                                       1.0, 'shape', 100, self.path)

    ### Provide observation
    def observe(self):

        # Always return the same observation
        return self.obs

    ### Take one step
    def step(self, act, ep):

        # Convert actions
        acc = self.convert_actions(act)

        # Generate new shape
        self.generate_shape(acc, ep)

        # Solve CFD with LBM
        drg, lft = solve(self.shape, self.path, ep)

        # Compute reward
        rwd = self.compute_reward(drg, lft)

        return rwd, acc

    ### Convert actions
    def convert_actions(self, flat_act):

        # Make new array for new shape
        act      = np.array(flat_act).reshape((self.n_mv_pts,self.vol_pt))
        acc      = act.copy()
        acc[:,:] = 0.0

        # Convert action to coordinates and add obstacles to problem
        for i in range(self.n_mv_pts):
            pt       = self.mv_pts[i]
            acc[i,:] = self.geom_transform(act[i,:], pt)

        return acc

    ### Generate shape
    def generate_shape(self, acc, ep):

        # Modify shape
        self.shape.modify_shape_from_field(acc,
                                           pts_list=self.mv_pts)
        self.shape.build()
        self.shape.write_csv(ep)
        self.shape.generate_image(plot_pts       = True,
                                  show_quadrants = True,
                                  min_radius     = self.min_rad,
                                  max_radius     = self.max_rad,
                                  index          = ep)

    ### Compute forward geometrical transformation for actions
    ### (u,v,w) assumed in [-1,1]**3
    def geom_transform(self, act, pt):

        u = act[0]
        v = act[1]
        w = act[2]

        radius = self.min_rad + 0.5*(self.max_rad-self.min_rad)*(u+1.0)
        dangle = (360.0/float(self.shape.n_control_pts))
        angle  = dangle*float(pt) + 0.5*v*dangle
        x      = radius*math.cos(math.radians(angle))
        y      = radius*math.sin(math.radians(angle))
        e      = self.min_edg + 0.5*(self.max_edg-self.min_edg)*(w+1.0)

        return np.array([x, y, e])

    ### Compute reward
    def compute_reward(self, drag, lift):

        # Drag is always <0 while lift changes sign
        reward =-lift/abs(drag)

        return reward

    ### Close environment
    def close(self):
        pass
