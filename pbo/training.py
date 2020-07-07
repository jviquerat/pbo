# Generic imports
import numpy as np

# Custom imports
from pbo.pbo      import *
from pbo.par_envs import *

########################
# Process training
########################
def launch_training(params, path, run):

    # Sanitize input
    if (params.n_cpu > params.n_ind):
        print('Error : n_cpu cannot exceed n_ind')
        exit()

    # Declare environment and agent
    env      = par_envs(params.env_name, params.n_cpu, path)
    n_params = env.n_params
    agent    = pbo(params, n_params)

    # Initialize parameters
    ep      = 0
    bst_acc = np.zeros(n_params)
    bst_rwd = -1.0e10

    # Loop over generations
    for gen in range(params.n_gen):

        # Handle n_cpu < n_ind
        size = params.n_ind//params.n_cpu
        rest = params.n_ind%params.n_cpu
        if (rest > 0): size += 1
        n_loop = params.n_cpu*np.ones((size), dtype=np.int16)
        if (rest > 0): n_loop[-1] = rest

        # Loop over individuals
        for i in range(size):

            # Make one iteration over all processes
            n            = n_loop[i]
            obs          = env.observe(n)
            act, mu, sig = agent.get_actions(obs, n)
            rwd, acc     = env.step(act, n, ep)
            agent.store_transition(obs, act, acc, rwd, mu, sig, n)

            # Store a few things
            for ind in range(n):
                agent.ep [ep] = ep
                agent.gen[ep] = gen
                ep           += 1

                # Store best reward
                if (rwd[ind] > bst_rwd):
                    bst_rwd = rwd[ind]
                    bst_acc = acc[ind]

        # Train network after one generation
        agent.compute_advantages()
        data = agent.train_networks()

        # Store for future file printing
        agent.store_learning_data(gen, ep, bst_rwd, bst_acc, data)

        # Write to files
        agent.write_learning_data(path, run)

        # Printings
        agent.print_generation(gen, bst_rwd)

    # Close environments
    env.close()
