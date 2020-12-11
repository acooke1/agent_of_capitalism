import os
import gym
import numpy as np
import tensorflow as tf
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class PPOModel(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        
        super(PPOModel, self).__init__()
        self.num_actions = num_actions

        self.epsilon = 0.1
        self.learning_rate = 2.5e-5
        self.GAMMA = 0.99
        self.LAMBDA = 0.95
        self.critic_discount = 0.4
        self.actor_discount = 1.0
        self.entropy_beta = 0.001
        self.non_zero = 1e-10

        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.hidden_size = 50
        self.state_size = state_size

        self.state_input = tf.keras.layers.Input(shape=self.num_actions)

        #ACTOR MODEL LAYERS

        self.actor1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.actor2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.out_actions_actor = tf.keras.layers.Dense(self.num_actions, activation='softmax')

        #CRITIC MODEL LAYERS

        self.critic1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        #self.critic2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.out_actions_critc = tf.keras.layers.Dense(1)

    def call(self, states):

        #output = self.state_input(states)
        states = np.asarray(states)
        output = self.actor1(states)
        output = self.actor2(output)
        output = self.out_actions_actor(output)

        return output

    def value_function(self, states):

        #output = self.state_input(states)
        states = np.asarray(states)
        output = self.critic1(states)
        #output = self.critic2(output)
        output = self.out_actions_critc(output)

        return output

    def loss(self, states, actions, discounted_rewards, previous_actions, oldpolicy_probs, gaes):

        values = self.value_function(states)
        oldpolicy_probs = tf.cast(oldpolicy_probs, dtype=tf.float32)
        newpolicy_probs = self.call(states)

        ratio = tf.math.exp(tf.subtract(tf.math.log(newpolicy_probs + 1e-10), tf.math.log(oldpolicy_probs + 1e-10)))
        p1 = ratio * gaes
        p2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * gaes
        actor_loss = -tf.reduce_mean(tf.minimum(p1, p2))
        critic_loss = tf.reduce_mean(tf.math.squared_difference(discounted_rewards, values))
        e = -tf.reduce_sum(newpolicy_probs * tf.math.log(tf.clip_by_value(newpolicy_probs, self.non_zero, 1)), axis=1)
        entropy = tf.reduce_mean(e, axis=0)
        #entropy = tf.reduce_mean(-(newpolicy_probs * tf.math.log(newpolicy_probs + 1e-10)))
        loss = self.critic_discount * critic_loss + self.actor_discount * actor_loss - self.entropy_beta * entropy
        #loss = - (actor_loss - self.constant_2 * critic_loss + self.constant_1 * entropy)
    
        return loss, newpolicy_probs

        
        # values = self.value_function(states)
        # zero = np.zeros((1,1))
        # next_state_values = np.append(values.numpy(), zero, axis=0)
        # advantages, deltas = get_gaes(discounted_rewards, values, next_state_values)
        # gaes = np.asarray(advantages)
        # newpolicy_probs = self.call(states)
        # action_probs = newpolicy_probs[-1:]
        # old_action_probs = oldpolicy_probs[-1:]
        # ratio = tf.exp(tf.math.log(action_probs + self.non_zero) - tf.math.log(old_action_probs + self.non_zero))
        # clipped_ratio = tf.clip_by_value(ratio, clip_value_min=1 - self.epsilon, clip_value_max=1 + self.epsilon) * advantages
        # actor_loss = -tf.reduce_mean(tf.minimum(tf.multiply(gaes, ratio), tf.multiply(gaes, clipped_ratio)))
        # #print((self.GAMMA * next_state_values).shape, " ", values.shape)
        # critic_loss = tf.reduce_mean(tf.math.squared_difference(discounted_rewards, values))
        # #critic_loss = tf.reduce_mean(tf.math.squared_difference(discounted_rewards + self.GAMMA * next_state_values, values))
        # e = -tf.reduce_sum(newpolicy_probs * tf.math.log(tf.clip_by_value(newpolicy_probs, self.non_zero, 1)), axis=1)
        # entropy = tf.reduce_mean(e, axis=0)
        # loss = - (actor_loss - self.actor_discount * critic_loss + self.critic_discount * entropy)
        # #print(newpolicy_probs)
        # return loss, newpolicy_probs
