# Generic imports
import sys
import numpy as np

# Custom imports
from pbo import *

########################
# Process training
########################
def launch_training(params, path, run):

    # Declare environment and agent
    sys.path.append(os.path.join(sys.path[0],'envs'))
    module    = __import__(params.env_name)
    env_build = getattr(module, params.env_name)
    env       = env_build()
    n_params  = env.n_params
    agent     = pbo(params, n_params)

    # Initialize parameters
    ep      = 0
    bst_acc = np.zeros(n_params)
    bst_rwd = -1.0e10

    # Loop over generations
    for gen in range(params.n_gen):

        # Printings
        agent.print_generation(gen)

        # Loop over individuals
        for ind in range(params.n_ind):

            # Make one iteration
            obs          = env.observe()
            act, mu, sig = agent.get_actions(obs)
            rwd, acc     = env.step(act)
            agent.store_transition(obs, act, acc, rwd, mu, sig)

            # Store a few things
            agent.ep [ep] = ep
            agent.gen[ep] = gen

            if (rwd > bst_rwd):
                bst_rwd = rwd
                bst_acc = acc

            # Update global index
            ep += 1

        # Train network after one generation
        agent.compute_advantages()
        data = agent.train_networks()

        # Store for future file printing
        agent.store_learning_data(gen, ep, bst_rwd, bst_acc, data)

    # Write to files
    agent.write_learning_data(path, run)
