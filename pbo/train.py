# Generic imports
import numpy as np

# Custom imports
from pbo.pbo      import *
from pbo.par_envs import *

########################
# Process training
########################
def train(params, output_path, env_path, run):

    # Sanitize input
    if (params.n_cpu > params.n_ind):
        print('Error : n_cpu cannot exceed n_ind')
        exit()

    # Declare environment and agent
    env      = par_envs(params.env_name,
                        params.n_cpu,
                        output_path+'/'+str(run),
                        env_path)
    act_size = env.act_size
    obs_size = env.obs_size
    agent    = pbo(params, act_size, obs_size)

    # Initialize parameters
    ep      = 0
    bst_acc = np.zeros(act_size)
    bst_rwd = -1.0e1
    bst_ep  = 0

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
            n                = n_loop[i]
            obs              = env.observe(n)
            act, mu, sig, cr = agent.get_actions(obs, n)
            rwd, acc         = env.step(act, n, ep)
            agent.store_transition(obs, act, acc, rwd, mu, sig, cr, n)

            # Loop on individuals
            for ind in range(n):

                # Store best reward
                if (rwd[ind] > bst_rwd):
                    bst_ep  = ep
                    bst_rwd = rwd[ind]
                    bst_acc = acc[ind]

                # Store ep and gen
                agent.ep [ep] = ep
                agent.gen[ep] = gen
                ep           += 1

        # Train network after one generation
        agent.compute_advantages()
        data = agent.train_networks()

        # Store for future file printing
        agent.store_learning_data(gen, bst_ep, bst_rwd, bst_acc, data)

        # Write to files
        agent.write_learning_data(output_path, run)

        # Printings
        agent.print_generation(gen, bst_rwd)

    # Close environments
    env.close()
