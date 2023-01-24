# Generic imports
import numpy as np

# Custom imports
from pbo.src.factory  import *
from pbo.src.par_envs import *

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
    agent = pbo_factory.create(params.type,
                               params  = params,
                               act_dim = act_size,
                               obs_dim = obs_size)

    # Initialize parameters
    ep      = 0
    bst_acc = np.zeros(act_size)
    bst_rwd =-1.0e8
    bst_ep  = 0

    # Loop over generations
    for gen in range(params.n_gen):

        # Handle n_cpu < n_ind
        size = params.n_ind//params.n_cpu
        rest = params.n_ind%params.n_cpu
        if (rest > 0): size += 1
        n_loop = params.n_cpu*np.ones((size), dtype=np.int16)
        if (rest > 0): n_loop[-1] = rest

        # Initialize avg rwd for this generation
        avg_rwd = 0

        # Loop over individuals
        for i in range(size):

            # Make one iteration over all processes
            n        = n_loop[i]
            obs      = env.observe(n)
            act      = agent.get_actions(obs, n)
            rwd, acc = env.step(act, n, ep)
            agent.store_transition(obs, act, acc, rwd, n)

            # Loop on individuals
            for ind in range(n):

                # Store best reward
                if (rwd[ind] >= bst_rwd):
                    bst_ep  = ep
                    bst_rwd = rwd[ind]
                    bst_acc = acc[ind]
                
                # Store ep and gen
                agent.ep [ep] = ep
                agent.gen[ep] = gen
                ep           += 1

            # Store avg reward
            avg_rwd += np.sum(rwd)

        # Train network after one generation
        agent.compute_advantages()
        agent.train_networks()

        # Store for future file printing
        agent.store_learning_data(gen, bst_ep, bst_rwd, bst_acc, avg_rwd/params.n_ind)

        # Write to files
        agent.write_learning_data(output_path, run)

        # Printings
        agent.print_generation(gen, bst_ep, bst_rwd[0])

    # Close environments
    env.close()
