# Generic imports
import os
import numpy as np

# Custom imports
from params      import *
from environment import *
from pbo         import *

########################
# Process training
########################
def launch_training():

    # Declare environement and agent
    env   = env_opt(init_obs, n_params, x_min, x_max, y_min, y_max)
    agent = pbo(pdf, loss, n_params, n_params, n_gen, n_ind, lr,
                mu_epochs, sg_epochs, clip_grd, sg_batch,
                clip_adv, clip_pol, mu_arch, sg_arch)

    # Initialize parameters
    episode = 0
    bst_cac = np.zeros(n_params)
    bst_rwd = -1.0e10

    # Loop over generations
    for gen in range(n_gen):

        # Printings
        if (gen == n_gen-1): end = '\n'
        if (gen != n_gen-1): end = '\r'
        print('#   Generation #'+str(gen), end=end)

        # Loop over individuals
        for ind in range(n_ind):

            # Make one iteration
            obs          = env.reset()
            act, mu, sig = agent.get_actions(obs)
            rwd, cac     = env.step(act)
            agent.store_transition(obs, act, cac, rwd, mu, sig)

            # Store a few things
            agent.ep [episode] = episode
            agent.gen[episode] = gen

            if (rwd > bst_rwd):
                bst_rwd = rwd
                bst_cac = cac

            # Update global index
            episode += 1

        # Store a few things
        agent.bst_gen[gen] = gen
        agent.bst_ep [gen] = episode
        agent.bst_rwd[gen] = bst_rwd
        agent.bst_cac[gen] = bst_cac

        # Train network after one generation
        agent.compute_advantages()
        agent.train_networks()

    # Write to files
    filename = 'database.opt.dat'
    np.savetxt(filename, np.transpose([agent.gen,
                                       agent.ep,
                                       agent.cac[:,0],
                                       agent.cac[:,1],
                                       agent.rwd*(-1.0),
                                       np.zeros(n_gen*n_ind)]))

    filename = 'optimisation.dat'
    np.savetxt(filename, np.transpose([agent.bst_gen+1,
                                       agent.bst_ep,
                                       agent.bst_cac[:,0],
                                       agent.bst_cac[:,1],
                                       agent.bst_rwd*(-1.0)]))

########################
# Average training over multiple runs
########################
idx  = np.zeros((      n_gen), dtype=int)
cost = np.zeros((n_avg,n_gen), dtype=float)

for i in range(n_avg):
    print('### Avg run #'+str(i))
    launch_training()

    f         = np.loadtxt('optimisation.dat')
    idx       = f[:,0]
    cost[i,:] = f[:,4]

# Write to file
file_out = 'ppo_avg_data.dat'
avg      = np.mean(cost,axis=0)
std      = 0.5*np.std(cost,axis=0)

# Be careful about standard deviation plotted in log scale
log_avg  = np.log(avg)
log_std  = 0.434*std/avg
log_p    = log_avg+log_std
log_m    = log_avg-log_std
p        = np.exp(log_p)
m        = np.exp(log_m)

array    = np.transpose(np.stack((idx, avg, m, p)))
np.savetxt(file_out, array)
os.system('gnuplot -c plot_single.gnu')
os.system('gnuplot -c plot_std.gnu')
