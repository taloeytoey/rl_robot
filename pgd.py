""" Trains an agent with (stochastic) Policy Gradients. """
import tensorflow as tf
import numpy as np
import random

from flat_game import carmunk

class PolicyNetwork():
    """
    Policy Function approximator. 
    """

    def __init__(self, learning_rate, scope="policy_network"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(dtype=tf.float32, shape=[None, 3], name="state")
            self.action = tf.placeholder(dtype=tf.int32, shape=[None, ], name="action")
            self.reward = tf.placeholder(dtype=tf.float32, shape=[None, ], name="reward")

            # FC1
            fc1 = tf.layers.dense(
                inputs=self.state,
                units=16,
                activation=tf.nn.tanh,  # tanh activation
                name='FC1'
            )

            # FC2
            fc2 = tf.layers.dense(
                inputs=fc1,
                units=32,
                activation=tf.nn.tanh,  # tanh activation
                name='FC2'
            )

            # logits
            logits = tf.layers.dense(
                inputs=fc2,
                units=3,
                activation=None,
                name='FC3'
            )

            self.action_prob = tf.nn.softmax(logits)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.action)

            self.loss = tf.reduce_mean(neg_log_prob * self.reward)
            # train op
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess):
        return sess.run(self.action_prob, {self.state: state})

    def update(self, state, reward, action, sess):
        feed_dict = {self.state: state, self.reward: reward, self.action: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

# hyperparameters
learning_rate = 0.005
gamma = 0.99  # discount factor for reward
resume = False  # resume from previous checkpoint?
render = True  # render the graphic ?
max_episode_number = 1000  # how many episode we want to run ?
model_path = "_models/reinforce/model.ckpt"  # path for saving the model

def discount_rewards(r):
    """
    take 1D float array of rewards and compute discounted rewards (A_t)
    A_t = R_t + gamma^1 * R_t+1 + gamma^2 * R_t+2 + ... + gamma^(T-t)R_T;
    where T is the last time step of the episode

    :param r: float array of rewards (R_1, R_2, ..., R_T)
    :return: float array of discounted reward (A_1, A_2, ..., A_T)
    """

    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        # if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r




if __name__ == '__main__':

    state_list, action_list, reward_list = [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0

    policy_network = PolicyNetwork(learning_rate)

    # saver
    saver = tf.train.Saver()
    # session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())



    # create a new game instance
    env = carmunk.GameState()

