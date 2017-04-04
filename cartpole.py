 ## 
#!/usr/bin/env python3

#####################
## Initial Imports ##
#####################

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *

import sys

#########################################
## Take in Arguments From Command Line ##
#########################################

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-l', '--load-model', metavar='NPZ',
                    help='NPZ file containing model weights/biases')
args = parser.parse_args()

#####################
## Execution Stuff ##
#####################

# Make Gyn Enviorment
env = gym.make('CartPole-v0')

# See random ganerators
RNG_SEED=1
tf.set_random_seed(RNG_SEED)
env.seed(RNG_SEED)

# Parameters for single layer NN as policy network
hidden_size = 8
alpha = 0.01
TINY = 1e-8
gamma = 0.98

# Initialize NN weights for polity
weights_init = xavier_initializer(uniform=False)
relu_init = tf.constant_initializer(0.1)

# If loading of previous model is selected, do that, if not use initialized weights
if args.load_model:
    model = np.load(args.load_model)
    hw_init = tf.constant_initializer(model['hidden/weights'])
    hb_init = tf.constant_initializer(model['hidden/biases'])
    mw_init = tf.constant_initializer(model['mus/weights'])
    mb_init = tf.constant_initializer(model['mus/biases'])
    sw_init = tf.constant_initializer(model['sigmas/weights'])
    sb_init = tf.constant_initializer(model['sigmas/biases'])
else:
    hw_init = weights_init
    hb_init = relu_init
    mw_init = weights_init
    mb_init = relu_init
    sw_init = weights_init
    sb_init = relu_init

# Get the number of outputs of NN to corrosponse to the number of inputs to model
try:
    output_units = env.action_space.shape[0]
except AttributeError:
    output_units = env.action_space.n

# Number of inputs to the nerual net, this is the number of observation outputs
# The model have
input_shape = env.observation_space.shape[0]

# Number of inputs, for use in tensorflow
NUM_INPUT_FEATURES = 4

# Tensorflow input and output initialization
x = tf.placeholder(tf.float32, shape=(None, NUM_INPUT_FEATURES), name='x')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

# Layers of the policy network
hidden = fully_connected(
    inputs=x,
    num_outputs=hidden_size,
    activation_fn=tf.nn.relu,
    weights_initializer=hw_init,
    weights_regularizer=None,
    biases_initializer=hb_init,
    scope='hidden')

mus = fully_connected(
    inputs=hidden,
    num_outputs=output_units,
    activation_fn=tf.tanh,
    weights_initializer=mw_init,
    weights_regularizer=None,
    biases_initializer=mb_init,
    scope='mus')

sigmas = tf.clip_by_value(fully_connected(
    inputs=hidden,
    num_outputs=output_units,
    activation_fn=tf.nn.softplus,
    weights_initializer=sw_init,
    weights_regularizer=None,
    biases_initializer=sb_init,
    scope='sigmas'),
    TINY, 5)

# A list of all variables decleared for tensorflow
all_vars = tf.global_variables()

# Layers of network
pi = tf.contrib.distributions.Normal(mus, sigmas, name='pi')
pi_sample = tf.argmax(tf.reduce_mean(tf.tanh(pi.sample(), name='pi_sample'),0),0)

log_pi = pi.log_prob(y, name='log_pi')


Returns = tf.placeholder(tf.float32, name='Returns')
optimizer = tf.train.GradientDescentOptimizer(alpha)
train_op = optimizer.minimize(-1.0 * Returns * log_pi)

# Declear and initialzei a tf session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# numer of state to remember? and get maximum iteration for that model
MEMORY=25
MAX_STEPS = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

# Run the enviorment
track_returns = []
for ep in range(16384):
    obs = env.reset()

    G = 0
    ep_states = []
    ep_actions = []
    ep_rewards = [0]
    done = False
    t = 0
    I = 1
    while not done:
        ep_states.append(obs)
        env.render()
        action =  sess.run([pi_sample], feed_dict={x:[obs]})[0]
        ep_actions.append([action])
        obs, reward, done, info = env.step(action)
        ep_rewards.append(reward * I)
        G += reward * I
        I *= gamma

        t += 1
        if t >= MAX_STEPS:
            break

    if not args.load_model:
        returns = np.array([G - np.cumsum(ep_rewards[:-1])]).T
        index = ep % MEMORY
        
        
        _ = sess.run([train_op],
                    feed_dict={x:np.array(ep_states),
                                y:np.array(ep_actions),
                                Returns:returns })

    track_returns.append(G)
    track_returns = track_returns[-MEMORY:]
    mean_return = np.mean(track_returns)
    print("Episode {} finished after {} steps with return {}".format(ep, t, G))
    print("Mean return over the last {} episodes is {}".format(MEMORY,
                                                               mean_return))


    with tf.variable_scope("mus", reuse=True):
        print("incoming weights for the mu's from the first hidden unit:", sess.run(tf.get_variable("weights"))[0,:])


sess.close()