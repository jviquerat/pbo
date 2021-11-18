# Generic imports
import numpy as np

# Custom imports
from utils.shapes           import *
from utils.shape_opt_solver import *
from envs.base_env          import *

###############################################
### Environment for shape optimization
class shape_opt(base_env):

    ### Create object
    def __init__(self, path):

        # Fill structure
        self.name     = 'shape_env'
        self.n_pts    = 4
        self.n_mv_pts = 3
        self.vol_pt   = 3
        self.act_size = self.vol_pt*self.n_mv_pts
        self.obs_size = self.act_size
        self.obs      = np.zeros(self.obs_size)
        self.min_rad  = 0.05
        self.max_rad  = 0.5
        self.min_edg  = 0.0
        self.max_edg  = 1.0
        self.mv_pts   = [0,1,3]
        self.x_min    =-2.0
        self.x_max    = 10.0
        self.y_min    =-2.0
        self.y_max    = 2.0
        self.Re       = 40.0
        self.path     = path
        self.lat_path = self.path+'/lattice'
        self.shape    = generate_shape(self.n_pts, [0.0,0.0], 'cylinder',
                                       1.0, 'shape', 100, self.path)

    ### Take one step
    def step(self, act, ep):

        # Convert actions
        acc = self.convert_actions(act)

        # Generate new shape
        self.update_shape(acc, ep)

        # Solve CFD with LBM
        drg, lft, _ = cfd_solve(self.x_min,    self.x_max,
                                self.y_min,    self.y_max,
                                self.Re,       self.shape,
                                self.lat_path, str(ep))

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

    ### Update shape
    def update_shape(self, acc, ep):

        # Modify shape
        self.shape.modify_shape_from_field(acc,
                                           pts_list=self.mv_pts)
        self.shape.build()
        self.shape.write_csv(ep)
        self.shape.generate_image(plot_pts       = True,
                                  show_quadrants = True,
                                  min_radius     = self.min_rad,
                                  max_radius     = self.max_rad,
                                  ep             = ep)

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
        reward = lift/abs(drag)

        return reward
