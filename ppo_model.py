import os
import gym
import numpy as np
import tensorflow as tf
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_gaes(rewards, state_values, next_state_values, GAMMA=0.99, LAMBDA=0.95):
    deltas = [r_t + GAMMA * next_v - v for r_t, next_v, v in zip(rewards, next_state_values, state_values)]
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):
        gaes[t] = gaes[t] + LAMBDA * GAMMA * gaes[t + 1]
    return gaes, deltas

class PPOModel(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        
        super(PPOModel, self).__init__()
        self.num_actions = num_actions

        self.epsilon = 0.2
        self.learning_rate = 2e-4
        self.GAMMA = 0.99
        self.LAMBDA = 0.95
        self.critic_discount = 0.5
        self.actor_discount = 1.0
        self.entropy_beta = 0.001
        self.non_zero = 1e-10

        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.hidden_size = 50
        self.hidden_size_first = 512
        self.hidden_size_second = 256
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

    def loss(self, states, actions, discounted_rewards, previous_actions, oldpolicy_probs):

        values = self.value_function(states)
        newpolicy_probs = self.call(states)
        zero = np.zeros((1,1))
        next_state_values = np.append(values.numpy(), zero, axis=0)
        advantages, deltas = get_gaes(discounted_rewards, values, next_state_values)
        shape_dif = newpolicy_probs.shape[0] - oldpolicy_probs.shape[0]
        correction = np.zeros((abs(shape_dif), self.num_actions))
        newpolicy_probs_fixed = newpolicy_probs
        oldpolicy_probs_fixed = oldpolicy_probs
        #if (shape_dif < 0):
        #    newpolicy_probs_fixed = tf.concat([newpolicy_probs_fixed, correction], axis=0)
        #elif (shape_dif > 0):
        #    oldpolicy_probs_fixed = tf.concat([oldpolicy_probs_fixed, correction], axis=0)
        ratio = tf.math.exp(tf.math.log(newpolicy_probs_fixed + 1e-10) - tf.math.log(oldpolicy_probs_fixed + 1e-10))
        p1 = ratio * advantages
        p2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
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
