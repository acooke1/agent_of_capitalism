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

        self.epsilon = 0.2
        self.learning_rate = 4e-4
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

    def loss(self, states, actions, discounted_rewards, oldpolicy_probs):

        values = self.value_function(states)
        oldpolicy_probs = tf.cast(oldpolicy_probs, dtype=tf.float32)
        newpolicy_probs = self.call(states)

        advantages = discounted_rewards - tf.squeeze(values)
        new_policy_log = tf.math.log(tf.gather_nd(newpolicy_probs, list(zip(np.arange(len(actions)), actions))))
        old_policy_log = tf.math.log(tf.gather_nd(oldpolicy_probs, list(zip(np.arange(len(actions)), actions))))
                    
        ratio = tf.math.exp(tf.subtract(new_policy_log, old_policy_log))
        #ratio = tf.math.exp(tf.subtract(tf.math.log(newpolicy_probs + 1e-10), tf.math.log(oldpolicy_probs + 1e-10)))
        p1 = ratio * advantages
        p2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -tf.reduce_mean(tf.minimum(p1, p2))
        critic_loss = tf.reduce_mean(tf.math.squared_difference(discounted_rewards, values))
        e = -tf.reduce_sum(newpolicy_probs * tf.math.log(tf.clip_by_value(newpolicy_probs, self.non_zero, 1)), axis=1)
        entropy = tf.reduce_mean(e, axis=0)
        #entropy = tf.reduce_mean(-(newpolicy_probs * tf.math.log(newpolicy_probs + 1e-10)))
        loss = self.critic_discount * critic_loss + self.actor_discount * actor_loss - self.entropy_beta * entropy
        #loss = - (actor_loss - self.constant_2 * critic_loss + self.constant_1 * entropy)
    
        return loss
