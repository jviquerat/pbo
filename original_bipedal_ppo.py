"""This is an Tensorflow 2.0 (Keras) implementation of a Open Ai's proximal policy optimization PPO algorithem for continuous action spaces.
Goal was to make it understanable yet not deviate from the original PPO idea: https://arxiv.org/abs/1707.06347 
Part of the code base is from https://github.com/liziniu/RL-PPO-Keras . However, the code there had errors
but mainly it did not use a GAE type reward and no entropy bonus system.
I gave my best to comment the code but I did not include a fundamental lecutre on the logic behind PPO. I highly 
recommend to watch these two videos to undestand what happens.
https://youtu.be/WxQfQW48A4A
https://youtu.be/5P7I-xPq8u8
The most complete explenation and also part of the code (i.e. Memory Class)
is from the open ai spinning up project: https://spinningup.openai.com/en/latest/algorithms/ppo.html
I did NOT test this, there might be errors. In a first attempt, the best score was somewhere around -70 for bipedap-walker 
which seems to show some leraning but not great learning.
TODO / Next steps:
1) Try some parameters to find a reasonably quick leraning agent. Currently does not converge or only very slowly.
2) try use tf.distribution to replace maual Probability Density and entropy calculations. 
3) Currently, the two outputs of actor (mu and sigma) are concatenated and then disassembled for the loss. Because the loss depends on both outputs at the same time (mu and sigma). I found this to be the only alternative to writing a custom train fuction with keras.function which seems not to work with TF 2.0 alpha. I should at least try to find a more elegant method.
4) read and implement tf.probability layers independant_normal - does this even make sense here?
"""

# %%
import numpy as np
import os
import time
import gym
from collections import deque
import tensorflow as tf
import tensorflow.keras as K
from scipy import signal


# %%
class MemoryPPO:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    This is adapted version of the Spinning Up Open Ai PPO code of the buffer.
    https://github.com/openai/spinningup/blob/master/spinup/algos/ppo/ppo.py
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        # a fucntion so that different dimensions state array shapes are all processed corecctly
        def combined_shape(length, shape=None):
            if shape is None:
                return (length,)
            return (length, shape) if np.isscalar(shape) else (length, *shape)
        # just empty arrays with appropriate sizes
        self.obs_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)  # states
        self.act_buf = np.zeros(combined_shape(
            size, act_dim), dtype=np.float32)  # actions
        # actual rwards from state using action
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # predicted values of state
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)  # gae advantewages
        self.ret_buf = np.zeros(size, dtype=np.float32)  # discounted rewards
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        example input: [x0, x1, x2] output: [x0 + discount * x1 + discount^2 * x2,  x1 + discount * x2, x2]
        """
        return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def store(self, obs, act, rew, val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val=0):
        """Finishes an episode of data collection by calculating the diffrent rewards and resetting pointers.
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(
            deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get_batch(self, batch_size):
        """simply retuns a randomized batch of batch_size from the data in memory 
        """
        # make a randlim list with batch_size numbers.
        pos_lst = np.random.randint(self.ptr, size=batch_size)
        return self.obs_buf[pos_lst], self.act_buf[pos_lst], self.adv_buf[pos_lst], self.ret_buf[pos_lst], self.val_buf[pos_lst]

    def clear(self):
        """Set back pointers to the beginning
        """
        self.ptr, self.path_start_idx = 0, 0


# %%
class Agent:
    def __init__(self, action_n, state_dim, TRAINING_BATCH_SIZE, TRAJECTORY_BUFFER_SIZE):
        """This initializes the agent object.
        Main interaction is the choose_action, store transition and train_network. 
        The agent only requires the state and action spaces to fuction, other than that it is pretty general
        and should be easy to adapt for other continuous envs.
        To understand what is happening, I recommend to look at the ppo_loss method and the build_actor method first.
        The training method itself is more or less only data preperation for calling the fit functions
        for actor and critic. But critic has a trivial loss, so all the PPO magic is in the ppo_loss function.
        """
        self.action_n = action_n
        self.state_dim = state_dim
        # CONSTANTS
        self.TRAINING_BATCH_SIZE = TRAINING_BATCH_SIZE
        self.TRAJECTORY_BUFFER_SIZE = TRAJECTORY_BUFFER_SIZE
        self.TARGET_UPDATE_ALPHA = 0.95
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.CLIPPING_LOSS_RATIO = 0.1
        self.ENTROPY_LOSS_RATIO = 0.001
        self.TARGET_UPDATE_ALPHA = 0.9
        self.NOISE = 1.0  # Exploration noise, for continous action space
        # create actor and critic neural networks
        self.critic_network = self._build_critic_network()
        self.actor_network = self._build_actor_network()
        # for the loss function, additionally "old" predicitons are required from before the last update.
        # therefore create another networtk. Set weights to be identical for now.
        self.actor_old_network = self._build_actor_network()
        self.actor_old_network.set_weights(self.actor_network.get_weights())
        # for getting an action (predict), the model requires it's ususal input, but advantage and old_prediction is only used for loss(training). So create dummys for prediction only
        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediciton = np.zeros((1, 2*self.action_n))
        # our transition memory buffer
        self.memory = MemoryPPO(
            self.state_dim, self.action_n, self.TRAJECTORY_BUFFER_SIZE)

    def ppo_loss(self, advantage, old_prediction):
        """The PPO custom loss.
        For explanation see for example:
        https://youtu.be/WxQfQW48A4A
        https://youtu.be/5P7I-xPq8u8
        Log Probability of  loss: (x-mu)²/2sigma² - log(sqrt(2*PI*sigma²))
        entropy of normal distribution: sqrt(2*PI*e*sigma²)
        params:
            :advantage: advantage, needed to process algorithm
            :old_predictioN: prediction from "old" network, needed to process algorithm
        returns:
            :loss: keras type loss fuction (not a value but a fuction with two parameters y_true, y_pred)
        TODO:
            probs = tf.distributions.Normal(mu,sigma)
            probs.sample #choses action
            probs.prob(action) #probability of action 
            https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/simply_PPO.py
        """

        def get_log_probability_density(network_output_prediction, y_true):
            """Sub-function to get the logarithmic probability density.
            expects the prediction (containing mu and sigma) and the true action y_true
            Formula for pdf and log-pdf see https://en.wikipedia.org/wiki/Normal_distribution
            """
            # the actor output contains mu and sigma concatenated. split them. shape is (batches,2xaction_n)
            mu_and_sigma = network_output_prediction
            mu = mu_and_sigma[:, 0:self.action_n]
            sigma = mu_and_sigma[:, self.action_n:]
            variance = K.backend.square(sigma)
            pdf = 1. / K.backend.sqrt(2. * np.pi * variance) * K.backend.exp(-K.backend.square(y_true - mu) / (2. * variance))
            log_pdf = K.backend.log(pdf + K.backend.epsilon())
            return log_pdf

        # refer to Keras custom loss function intro to understand why we define a funciton inside a function.
        # here y_true are the actions taken and y_pred are the predicted prob-distribution(mu,sigma) for each n in acion space
        def loss(y_true, y_pred):
            # First the probability density function.
            log_probability_density_new = get_log_probability_density(y_pred, y_true)
            log_probability_density_old = get_log_probability_density(old_prediction, y_true)
            # Calc ratio and the surrogates
            # ratio = prob / (old_prob + K.epsilon()) #ratio new to old
            ratio = K.backend.exp(log_probability_density_new-log_probability_density_old)
            surrogate1 = ratio * advantage
            clip_ratio = K.backend.clip(ratio, min_value=1 - self.CLIPPING_LOSS_RATIO, max_value=1 + self.CLIPPING_LOSS_RATIO)
            surrogate2 = clip_ratio * advantage
            # loss is the mean of the minimum of either of the surrogates
            loss_actor = - K.backend.mean(K.backend.minimum(surrogate1, surrogate2))
            # entropy bonus in accordance with move37 explanation https://youtu.be/kWHSH2HgbNQ
            sigma = y_pred[:, self.action_n:]
            variance = K.backend.square(sigma)
            loss_entropy = self.ENTROPY_LOSS_RATIO * K.backend.mean(-(K.backend.log(2*np.pi*variance)+1) / 2)  # see move37 chap 9.5
            # total bonus is all losses combined. Add MSE-value-loss here as well?
            return loss_actor + loss_entropy
        return loss

        

    def _build_actor_network(self):
        """builds and returns a compiled keras.model for the actor.
        There are 3 inputs. Only the state is for the pass though the neural net. 
        The other two inputs are exclusivly used for the custom loss function (ppo_loss).
        """
        # define inputs. Advantage and old_prediction are required to pass to the ppo_loss funktion
        state = K.layers.Input(shape=(self.state_dim[0],), name='state_input')
        advantage = K.layers.Input(shape=(1,), name='advantage_input')
        old_prediction = K.layers.Input(shape=(2*self.action_n,), name='old_prediction_input')
        # define hidden layers
        dense = K.layers.Dense(32, activation='relu', name='dense1')(state)
        dense = K.layers.Dense(32, activation='relu', name='dense2')(dense)
        # connect layers. In the continuous case the actions are not probabilities summing up to 1 (softmax)
        # but squshed numbers between -1 and 1 for each action (tanh). This represents the mu of a gaussian
        # distribution
        mu = K.layers.Dense(self.action_n, activation='tanh',name="actor_output_mu")(dense)
        #mu = 2 * muactor_output_layer_continuous
        # in addtion, we have a second output layer representing the sigma for each action
        sigma = K.layers.Dense(self.action_n, activation='softplus', name="actor_output_sigma")(dense)
        #sigma = sigma + K.backend.epsilon()
        # concat layers. The alterative would be to have two output heads but this would then require to make a custom
        # keras.function insead of the .compile and .fit routine adding more distraciton
        mu_and_sigma = K.layers.concatenate([mu, sigma])
        # make keras.Model
        actor_network = K.Model(
            inputs=[state, advantage, old_prediction], outputs=mu_and_sigma)
        # compile. Here the connection to the PPO loss fuction is made. The input placeholders are passed.
        actor_network.compile(
            optimizer='adam', loss=self.ppo_loss(advantage, old_prediction))
        # summary and return
        actor_network.summary()
        return actor_network

    def _build_critic_network(self):
        """builds and returns a compiled keras.model for the critic.
        The critic is a simple scalar prediction on the state value(output) given an state(input)
        Loss is simply mse
        """
        # define input layer
        state = K.layers.Input(shape=(self.state_dim[0],), name='state_input')
        # define hidden layers
        dense = K.layers.Dense(32, activation='relu', name='dense1')(state)
        dense = K.layers.Dense(32, activation='relu', name='dense2')(dense)
        # connect the layers to a 1-dim output: scalar value of the state (= Q value or V(s))
        V = K.layers.Dense(1, name="actor_output_layer")(dense)
        # make keras.Model
        critic_network = K.Model(inputs=state, outputs=V)
        # compile. Here the connection to the PPO loss fuction is made. The input placeholders are passed.
        critic_network.compile(optimizer='Adam', loss='mean_squared_error')
        # summary and return
        critic_network.summary()
        return critic_network

    def update_tartget_network(self):
        """Softupdate of the target network.
        In ppo, the updates of the 
        """
        alpha = self.TARGET_UPDATE_ALPHA
        actor_weights = np.array(self.actor_network.get_weights())
        actor_tartget_weights = np.array(self.actor_old_network.get_weights())
        new_weights = alpha*actor_weights + (1-alpha)*actor_tartget_weights
        self.actor_old_network.set_weights(new_weights)

    def choose_action(self, state, optimal=False):
        """chooses an action within the action space given a state. 
        The action is chosen by random with the weightings accoring to the probability
        params:
            :state: np.array of the states with state_dim length
            :optimal: if True, the agent will always give best action for state. 
                     This will cause no exploring! --> deactivate for learning, just for evaluation
        """
        assert isinstance(state, np.ndarray)
        assert state.shape == self.state_dim
        # reshape for predict_on_batch which requires 2d-arrays (batches,state_dims) but only one batch
        state = state.reshape(1,-1)
        # the probability list for each action is the output of the actor network given a state
        # output has shape (batchsize,2xaction_n)
        mu_and_sigma = self.actor_network.predict_on_batch(
            [state, self.dummy_advantage, self.dummy_old_prediciton])
        mu = mu_and_sigma[0, 0:self.action_n]
        sigma = mu_and_sigma[0, self.action_n:]
        # action is chosen by random with the weightings accoring to the probability
        if optimal:
            action = mu
        else:
            action = np.random.normal(loc=mu, scale=sigma, size=self.action_n)
        return action

    def train_network(self):
        """Train the actor and critic networks using GAE Algorithm.
        1. Get GAE rewards, s,a,
        2. get "old" precition (of target network)
        3. fit actor and critic network
        4. soft update target "old" network
        """

        # get randomized mini batches
        states, actions, gae_advantages, discounted_rewards, values = self.memory.get_batch(
            self.TRAINING_BATCH_SIZE)
        gae_advantages = gae_advantages.reshape(-1,1) #batches of shape (1,) required
        gae_advantages = K.utils.normalize(gae_advantages)  # optionally normalize
        # calc old_prediction. Required for actor loss.
        batch_old_prediction = self.get_old_prediction(states)
        # commit training
        self.actor_network.fit(
            x=[states, gae_advantages, batch_old_prediction], y=actions, verbose=0)
        self.critic_network.fit(
            x=states, y=discounted_rewards, epochs=1, verbose=0)
        # soft update the target network(aka actor_old).
        self.update_tartget_network()

    def store_transition(self, s, a, r):
        """Store the experiences transtions into memory object.
        """
        value = self.get_v(s).flatten()
        self.memory.store(s, a, r, value)

    def get_v(self, state):
        """Returns the value of the state.
        Basically, just a forward pass though the critic networtk
        """
        s = np.reshape(state, (-1, self.state_dim[0]))
        v = self.critic_network.predict_on_batch(s)
        return v

    def get_old_prediction(self, state):
        """Makes an prediction (an action) given a state on the actor_old_network.
        This is for the train_network --> ppo_loss
        """
        state = np.reshape(state, (-1, self.state_dim[0]))
        return self.actor_old_network.predict_on_batch([state, self.dummy_advantage, self.dummy_old_prediciton])


# %%
ENV_NAME = "BipedalWalker-v2"
EPOCHS = 1000
MAX_EPISODE_STEPS = 2000
# train at the end of each epoch for simplicity. Not necessarily better.
TRAJECTORY_BUFFER_SIZE = MAX_EPISODE_STEPS
BATCH_SIZE = 256
RENDER_EVERY = 10


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    agent = Agent(env.action_space.shape[0], env.observation_space.shape,
                  BATCH_SIZE, TRAJECTORY_BUFFER_SIZE)
    for epoch in range(EPOCHS):
        s = env.reset()
        r_sum = 0
        for t in range(MAX_EPISODE_STEPS):
            # sometimes render
            #if epoch % RENDER_EVERY == 0:
            #    env.render()
            # get action from agent given state
            a = agent.choose_action(s)
            # get s_,r,done
            s_, r, done, _ = env.step(a)
            # store transitions to agent.memory
            agent.store_transition(s, a, r)
            r_sum += r
            if done or (t == MAX_EPISODE_STEPS-1):
                # predict critic for s_ (value of s_)
                last_val = r if done else agent.get_v(s_)
                # do the discounted_rewards and GAE calucaltions
                agent.memory.finish_path(last_val)
                break
        agent.train_network()
        agent.memory.clear()
        if epoch % 10 == 0:
            print(f"Episode:{epoch}, step:{t}, r_sum:{r_sum}")
