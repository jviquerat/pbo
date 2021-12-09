# Generic imports
import math
import time
import warnings
warnings.filterwarnings("error")

# Custom imports
from utils.shapes  import *
from utils.lattice import *

###############################################
### LBM solver
###############################################

# Generic solving function
def cfd_solve(x_min, x_max, y_min, y_max, Re,
              shape, path, suffix):

    # Free parameters
    # L_lbm correponds to y length
    # u_lbm corresponds to max velocity
    Re_lbm      = Re
    u_lbm       = 0.1
    L_lbm       = 200

    # Deduce other parameters
    obs_size    = 1.0
    Cs          = 1.0/math.sqrt(3.0)
    ny          = L_lbm
    u_avg       = 2.0*u_lbm/3.0
    D_lbm       = math.floor(ny*obs_size/(y_max-y_min))
    nu_lbm      = u_avg*D_lbm/Re_lbm
    tau_lbm     = 0.5 + nu_lbm/(Cs**2)
    rho_lbm     = 1.0
    dt          = Re_lbm*nu_lbm/L_lbm**2
    dx          = (y_max-y_min)/ny
    dy          = dx
    nx          = math.floor(ny*(x_max-x_min)/(y_max-y_min))

    # Other parameters
    output_freq = 5000
    dpi         = 200
    IBB         = True
    stop        = 'obs'
    obs_cv_ct   = 1.0e-2
    obs_cv_nb   = 1000
    t_max       = 0.02
    it_max      = math.floor(t_max/dt)
    sigma       = math.floor(10*nx)

    # Initialize lattice
    lat = Lattice(nx          = nx,
                  ny          = ny,
                  dx          = dx,
                  dt          = dt,
                  tau_lbm     = tau_lbm,
                  Re_lbm      = Re_lbm,
                  u_lbm       = u_lbm,
                  L_lbm       = D_lbm,
                  nu_lbm      = nu_lbm,
                  rho_lbm     = rho_lbm,
                  x_min       = x_min,
                  x_max       = x_max,
                  y_min       = y_min,
                  y_max       = y_max,
                  dpi         = dpi,
                  IBB         = IBB,
                  stop        = stop,
                  t_max       = t_max,
                  it_max      = it_max,
                  obs_cv_ct   = obs_cv_ct,
                  obs_cv_nb   = obs_cv_nb,
                  path        = path,
                  suffix      = suffix)

    # Add obstacle
    lat.add_obstacle(shape.curve_pts, 1)
    lat.generate_image()

    # Initialize fieldcs
    lat.set_inlet_poiseuille(u_lbm, rho_lbm, 0, sigma)
    lat.u[:,np.where(lat.lattice > 0.0)] = 0.0

    # Set initial distributions
    lat.equilibrium()
    lat.g = lat.g_eq.copy()

    # Solve
    while (lat.compute):

        # Progressively impose Poiseuille
        lat.set_inlet_poiseuille(u_lbm, rho_lbm, lat.it, sigma)

        # Compute macroscopic fields
        lat.macro()

        # Output field
        lat.output_fields(lat.it,
                          output_freq,
                          u_norm   = True,
                          u_stream = False)

        # Compute equilibrium state
        lat.equilibrium()

        # Collisions
        lat.collision_stream()

        # Boundary conditions
        lat.bounce_back_obstacle(0)
        lat.zou_he_bottom_wall_velocity()
        lat.zou_he_left_wall_velocity()
        lat.zou_he_top_wall_velocity()
        lat.zou_he_right_wall_pressure()
        lat.zou_he_bottom_left_corner()
        lat.zou_he_top_left_corner()
        lat.zou_he_top_right_corner()
        lat.zou_he_bottom_right_corner()

        # Compute drag/lift
        drag, lift = lat.drag_lift(0, rho_lbm, u_avg, D_lbm)
        lat.add_buff(drag, lift, lat.it)

        # Check stopping criterion
        lat.check_stop()

    # Finalize
    success          = True
    avg_drag, growth = lat.drag_buff.mv_avg()
    avg_lift, growth = lat.lift_buff.mv_avg()
    if (avg_drag > 1.0e2): success = False

    return avg_drag, avg_lift, success
