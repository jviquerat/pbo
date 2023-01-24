# Generic imports
import os
import sys
import numpy           as np
import multiprocessing as mp

###############################################
### A wrapper class for parallel environments
class par_envs:
    def __init__(self, env_name, n_cpu, output_path, env_path):

        # Init pipes and processes
        self.n_cpu   = n_cpu
        self.p_pipes = []
        self.proc    = []

        # Start environments
        for cpu in range(n_cpu):
            p_pipe, c_pipe = mp.Pipe()
            name           = str(cpu)
            process = mp.Process(target = worker,
                                 name   = name,
                                 args   = (env_name, name,
                                           c_pipe, p_pipe,
                                           output_path, env_path))

            self.p_pipes.append(p_pipe)
            self.proc.append(process)

            process.daemon = True
            process.start()
            c_pipe.close()

        # Handle nb of parameters
        self.act_size, self.obs_size = self.get_dims()

    # Reset all environments
    def observe(self, n):

        # Send
        for pipe in range(n):
            self.p_pipes[pipe].send(('observe',None,None))

        # Receive
        results = np.array([])
        for pipe in range(n):
            results = np.append(results, self.p_pipes[pipe].recv())

        results = np.reshape(results, (n,self.obs_size))

        return results

    # Get environment dimensions
    def get_dims(self):

        # Send
        self.p_pipes[0].send(('get_dims',None,None))

        # Receive
        act_size, obs_size = self.p_pipes[0].recv()

        return act_size, obs_size

    # Close
    def close(self):

        for p in self.proc:
            p.terminate()
            p.join()

    # Take one step in all environments
    def step(self, actions, n, ep):

        # Send
        for pipe in range(n):
            self.p_pipes[pipe].send(('step', actions[pipe], ep+pipe))
            
        # Receive
        rwd = np.array([])
        acc = np.array([])
        for pipe in range(n):
            r, a  = self.p_pipes[pipe].recv()
            rwd   = np.append(rwd, r)
            acc   = np.append(acc, a)

        # quick fix
        temp_size = self.act_size
        if isinstance(self.act_size,list):
            temp_size = len(self.act_size)

        acc = np.reshape(acc, (n,temp_size))
        rwd = np.reshape(rwd, (n,1))

        return rwd, acc

# Target function for process
def worker(env_name, name, pipe, p_pipe, output_path, env_path):

    # Build environment
    sys.path.append(env_path)
    module    = __import__(env_name)
    env_build = getattr(module, env_name)
    env       = env_build(output_path)
    p_pipe.close()

    # Execute tasks
    try:
        while True:
            # Receive command
            command, data, ep = pipe.recv()

            # Execute command
            if command == 'observe':
                obs = env.observe()
                pipe.send(obs)
            if command == 'step':
                rwd, acc = env.step(data, ep)
                pipe.send((rwd, acc))
            if (command == 'get_dims'):
                act_size = env.act_size # remove int()
                obs_size = int(env.obs_size)
                pipe.send((act_size, obs_size))
            if command == 'close':
                pipe.send(None)
                break
    finally:
        env.close()
