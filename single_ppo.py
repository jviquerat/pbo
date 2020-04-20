# Generic imports
import os
import gym
import warnings
import numpy as np

# Import tensorflow and filter warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow       as tf
import tensorflow.keras as K
tf.logging.set_verbosity(tf.logging.FATAL)

# Custom imports
from environment import *

###############################################
### Class ppo_agent
### A single-step PPO agent for optimization
class ppo_agent:
    def __init__(self,
                 act_dim, obs_dim, n_gen, n_ind,
                 clip, entropy, learn_rate, actor_epochs):

        # Initialize from arguments
        self.act_dim    = act_dim
        self.obs_dim    = obs_dim
        self.n_gen      = n_gen
        self.n_ind      = n_ind
        self.size       = self.n_gen*self.n_ind

        self.clip       = clip
        self.entropy    = entropy
        self.learn_rate = learn_rate
        self.actor_epochs = actor_epochs


        # Build actors
        self.actor     = self.build_actor()
        self.old_actor = self.build_actor()
        self.old_actor.set_weights(self.actor.get_weights())

        # Generate dummy inputs for custom loss
        self.dummy_adv  = np.zeros((1, 1))
        self.dummy_pred = np.zeros((1, 2*self.act_dim))

        # Storing buffers
        self.idx       = 0

        self.gen       = np.zeros( self.size,                 dtype=np.int32)
        self.ep        = np.zeros( self.size,                 dtype=np.int32)
        self.obs       = np.zeros((self.size,  self.obs_dim), dtype=np.float32)
        self.act       = np.zeros((self.size,  self.act_dim), dtype=np.float32)
        self.cact      = np.zeros((self.size,  self.act_dim), dtype=np.float32)
        self.rwd       = np.zeros( self.size,                 dtype=np.float32)
        self.val       = np.zeros( self.size,                 dtype=np.float32)

        self.best_cact = np.zeros((self.n_gen, self.act_dim), dtype=np.float32)
        self.best_rwd  = np.zeros( self.n_gen,                dtype=np.float32)
        self.best_gen  = np.zeros( self.n_gen,                dtype=np.int32)
        self.best_ep   = np.zeros( self.n_gen,                dtype=np.int32)

        self.adv       = np.zeros( self.size,                 dtype=np.float32)

    # Get batch of observations, actions and rewards
    def get_batch(self, n_batch):

        # Starting and ending indices based on the required size of batch
        start = max(0,self.idx - n_batch)
        end   = self.idx

        return self.obs[start:end], self.act[start:end], self.adv[start:end]

    # Custom ppo loss
    def ppo_loss(self, adv, old_act):

        # Log prob density function
        def log_prob_density(prediction, y_true):

            # Compute log prob density
            mu       = prediction[:, 0:self.act_dim]
            sigma    = prediction[:, self.act_dim:]
            variance = K.backend.square(sigma)
            factor   = 1.0/K.backend.sqrt(2.*np.pi*variance)
            pdf      = factor*K.backend.exp(-K.backend.square(y_true - mu)/(2.0*variance))
            log_pdf  = K.backend.log(pdf + K.backend.epsilon())

            return log_pdf

        def loss(y_true, y_pred):

            # Get the log prob density
            log_prob_density_new = log_prob_density(y_pred,  y_true)
            log_prob_density_old = log_prob_density(old_act, y_true)

            # Compute actor loss following Schulman
            ratio      = K.backend.exp(log_prob_density_new - log_prob_density_old)
            surrogate1 = ratio*adv
            clip_ratio = K.backend.clip(ratio,
                                        min_value=1 - self.clip,
                                        max_value=1 + self.clip)
            surrogate2 = clip_ratio*adv
            loss_actor =-K.backend.mean(K.backend.minimum(surrogate1, surrogate2))

            # Compute entropy loss
            sigma        = y_pred[:, self.act_dim:]
            variance     = K.backend.square(sigma)
            loss_entropy = K.backend.mean(-(K.backend.log(2.0*np.pi*variance)+1.0)/2.0)

            # Total loss
            return loss_actor + self.entropy*loss_entropy

        return loss

    # Build actor network using keras
    def build_actor(self):

        # Input layers
        # Forward network pass only requires observation
        # Advantage and old_action are only used in custom loss
        obs     = K.layers.Input(shape=(self.obs_dim,)  )
        adv     = K.layers.Input(shape=(1,),            )
        old_act = K.layers.Input(shape=(2*self.act_dim,))

        # Use orthogonal layers initialization
        init_1  = K.initializers.Orthogonal(gain=0.1, seed=None)
        init_2  = K.initializers.Orthogonal(gain=0.1, seed=None)

        # Dense layer, then one branch for mu and one for sigma
        dense     = K.layers.Dense(2,
                                   activation         = 'relu',
                                   kernel_initializer = init_1)(obs)
        mu        = K.layers.Dense(self.act_dim,
                                   activation         = 'linear',
                                   kernel_initializer = init_2)(dense)
        sigma     = K.layers.Dense(self.act_dim,
                                   activation         = 'softplus',
                                   kernel_initializer = init_2)(dense)

        # Concatenate outputs
        outputs   = K.layers.concatenate([mu, sigma])

        # Generate actor
        actor     = K.Model(inputs  = [obs, adv, old_act],
                            outputs = outputs)
        optimizer = K.optimizers.Adam(lr = self.learn_rate)
        actor.compile(optimizer = optimizer,
                      loss      = self.ppo_loss(adv, old_act))

        return actor

    def update_target_network(self):
        #alpha = self.TARGET_UPDATE_ALPHA
        actor_weights = self.actor.get_weights()
        #actor_target_weights = self.actor_old_network.get_weights()
        #new_weights = alpha*actor_weights + (1-alpha)*actor_target_weights
        new_weights = actor_weights
        self.old_actor.set_weights(new_weights)

    # Get actions from network
    def get_actions(self, state):

        # Reshape state
        state   = state.reshape(1,self.obs_dim)

        # Predict means and deviations
        # The two last parameters are dummy arguments: they are
        # only required for the custom loss used for training
        outputs = self.actor.predict([state, self.dummy_adv, self.dummy_pred])
        mu      = outputs[0,            0:self.act_dim]
        sigma   = outputs[0, self.act_dim:            ]

        # Draw action from normal law defined by mu and sigma
        actions = np.zeros(self.act_dim)
        for i in range(self.act_dim):
            actions[i] = np.random.normal(loc=mu[i], scale=sigma[i])

        return actions

    # Train actor network
    def train_network(self, n_batch):

        # Get batch
        obs, act, adv = self.get_batch(n_batch)

        # Compute old action
        old_act = self.get_old_actions(obs)

        # Train network
        self.actor.fit(x       = [obs, adv, old_act],
                       y       = act,
                       epochs  = self.actor_epochs,
                       verbose = 0)

        # soft update the target network(aka actor_old).
        self.update_target_network()

    # Store transitions into buffer
    def store_transition(self, obs, act, cact, rwd):

        # Fill buffers
        self.obs [self.idx] = obs
        self.act [self.idx] = act
        self.cact[self.idx] = cact
        self.rwd [self.idx] = rwd

        # Update index
        self.idx           += 1

    # Compute advantages
    def compute_advantages(self, n_batch):

        # Start and end indices of last generation
        start        = max(0,self.idx - n_batch)
        end          = self.idx

        # Compute normalized advantage
        avg_rwd      = np.mean(self.rwd[start:end])
        std_rwd      = np.std( self.rwd[start:end])
        self.adv[:]  = (self.rwd[:] - avg_rwd)/(std_rwd + 1.0e-7)

    # Get actions from previous policy
    def get_old_actions(self, state):

        # This is a batch of states, unlike in get_actions routine
        state   = state.reshape(self.n_ind, self.obs_dim)
        outputs = self.old_actor.predict_on_batch([state,
                                                   self.dummy_adv,
                                                   self.dummy_pred])

        return outputs



n_params = 2
init_obs = np.zeros(n_params)
x_min =-5.0
x_max = 5.0
y_min =-5.0
y_max = 5.0

n_gen   = 30
n_ind   = 6
n_batch = n_ind

clip = 0.5
entropy = 0.01
learn_rate = 5.0e-3
actor_epochs = 64

n_avg = 10


#env = gym.make(ENV_NAME)
def launch_training():
    env   = env_opt(init_obs, n_params, x_min, x_max, y_min, y_max)
    agent = ppo_agent(n_params, n_params, n_gen, n_ind,
                      clip, entropy, learn_rate, actor_epochs)

    episode = 0
    best_cact = np.zeros(n_params)
    best_rwd = -1.0e10

    for gen in range(n_gen):
        for ind in range(n_ind):

            s     = env.reset()
            a     = agent.get_actions(s)
            r, ca = env.step(a)
            agent.store_transition(s, a, ca, r)


            agent.ep [episode] = episode
            agent.gen[episode] = gen


            if (r > best_rwd):
                best_rwd  = r
                best_cact = ca

            episode += 1

        agent.best_gen [gen] = gen
        agent.best_ep  [gen] = episode
        agent.best_rwd [gen] = best_rwd
        agent.best_cact[gen] = best_cact

        print('### Generation #'+str(gen), end='\r')

        agent.compute_advantages(n_batch)

        # Train network
        agent.train_network(n_batch)
        #agent.memory.clear()

    # Outputs
    filename = 'database.opt.dat'
    np.savetxt(filename, np.transpose([agent.gen,
                                       agent.ep,
                                       agent.cact[:,0],
                                       agent.cact[:,1],
                                       agent.rwd*(-1.0),
                                       np.zeros(n_gen*n_ind)]))

    filename = 'optimisation.dat'
    np.savetxt(filename, np.transpose([agent.best_gen+1,
                                       agent.best_ep,
                                       agent.best_cact[:,0],
                                       agent.best_cact[:,1],
                                       agent.best_rwd*(-1.0)]))



if __name__ == "__main__":
    idx     = np.zeros((      n_gen), dtype=int)
    cost    = np.zeros((n_avg,n_gen), dtype=float)

    for i in range(n_avg):
        print('#   Avg run #'+str(i))
        launch_training()

        f         = np.loadtxt('optimisation.dat')
        idx       = f[:,0]
        cost[i,:] = f[:,4]

    # Write to file
    file_out = 'ppo_avg_data.dat'
    avg      = np.mean(cost,axis=0)
    std      = 0.5*np.std (cost,axis=0)

    log_avg  = np.log(avg)
    log_std  = 0.434*std/avg
    log_p    = log_avg+log_std
    log_m    = log_avg-log_std
    p        = np.exp(log_p)
    m        = np.exp(log_m)

    array    = np.transpose(np.stack((idx, avg, m, p)))
    np.savetxt(file_out, array)
