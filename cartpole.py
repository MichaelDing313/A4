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
alpha = 0.001
TINY = 1e-8
gamma = 0.99

# Initialize NN weights for policy
weights_init = xavier_initializer(uniform=False)
relu_init = tf.constant_initializer(0.1)

# If loading of previous model is selected, do that, if not use initialized weights
if args.load_model:
    model = np.load(args.load_model)
    sw_init = tf.constant_initializer(model['sigmas/weights'])
    sb_init = tf.constant_initializer(model['sigmas/biases'])
else:
    sw_init = weights_init
    sb_init = relu_init

# Number of inputs to the nerual net, this is the number of observation outputs
# The model have
input_shape = env.observation_space.shape[0]

# Tensorflow input and output initialization
# Input features: 4, output features 2
x = tf.placeholder(tf.float32, shape=(None, 4), name='x')
y = tf.placeholder(tf.float32, shape=(None, 2), name='y')

# Olny one lay network is required since the system is quite simple
sigmas = tf.clip_by_value(fully_connected(
    inputs=x,
    num_outputs=2,
    activation_fn=tf.nn.sigmoid,
    weights_initializer=sw_init,
    weights_regularizer=None,
    biases_initializer=sb_init,
    scope='sigmas'),
    TINY, 5)

# A list of all variables decleared for tensorflow
all_vars = tf.global_variables()

# Layers of network
# Use bernulie distribution here instead since actions are descrete
pi = tf.contrib.distributions.Bernoulli(p = sigmas, name='pi')
pi_sample = pi.sample()

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
plt_returns = []
plt_avg_returns = []
plt_sw = []
plt_x = []


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
        action =  sess.run([pi_sample], feed_dict={x:[obs]})[0][0]
    
        obs, reward, done, info = env.step(np.argmax(list(action)))
        
        ep_actions.append(action)
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
    plt_avg_returns.append(mean_return)
    plt_returns.append(G)

    with tf.variable_scope('sigmas', reuse=True):
        plt_sw.append(sess.run(tf.get_variable('weights')))
        #plt_sb.append(sess.run(tf.get_variable('bias')))
    print("Episode {} finished after {} steps with return {}".format(ep, t, G))
    print("Mean return over the last {} episodes is {}".format(MEMORY,
                                                               mean_return))

sess.close()

try:
    plt.figure()
    plt.plot(plot_x, plot_train, '-g', label='Training')
    plt.plot(plot_x, plot_vali, '-r', label='Validation')
    plt.plot(plot_x, plot_test, '-b', label='Test')
    plt.xlabel("Training Iterations")
    plt.ylabel("Accuracy")
    plt.title("Traning Curve for Single Hidden Layer NN, lam = {}".format(lam))
    plt.legend(loc='bottom right')
    plt.show()
except:
    print("plot fail")
        