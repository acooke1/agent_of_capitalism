import os
import gym
import numpy as np
import tensorflow as tf
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.backend.set_floatx('float64')

clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001
gamma = 0.99
lmbda = 0.95
num_actions = 4

class PPOModel(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        
        super(PPOModel, self).__init__()
        self.num_actions = num_actions

        self.epsilon = 0.2
        self.learning_rate = 2.5e-8
        self.GAMMA = 0.99
        self.LAMBDA = 0.95
        self.critic_discount = 0.4
        self.actor_discount = 1.0
        self.entropy_beta = 0.001
        self.non_zero = 1e-10
        self.eAdam = 1e-4

        self.mini_batch_size = 16
        self.ppo_epochs = 10

        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        self.hidden_size = 32
        self.state_size = state_size

        #ACTOR MODEL LAYERS

        self.actor1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.actor2 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.out_actions_actor = tf.keras.layers.Dense(self.num_actions, activation='softmax')

        #CRITIC MODEL LAYERS

        self.critic1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.out_actions_critc = tf.keras.layers.Dense(1)

    def call(self, states):

        states = np.asarray(states)
        output = self.actor1(states)
        output = self.actor2(output)
        output = self.out_actions_actor(output)

        return output

    def value_function(self, states):

        states = np.asarray(states)
        output = self.critic1(states)
        output = self.out_actions_critc(output)

        return output
    
    def ppo_iter(self, states, actions, actions_probs, returns, advantage, rewards):
        batch_size = states.shape[0]
        for _ in range(batch_size // self.mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], actions_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :], rewards[rand_ids, :]

    def loss(self, states, actions, rewards, actions_probs, returns, advantages, log_probs, actions_onehot):
        episode_loss = []
        for _ in range(self.ppo_epochs):
            for state, action, old_policy_probs, return_, advantage, reward in self.ppo_iter(states, actions, actions_probs, returns, advantages, rewards):
                with tf.GradientTape() as tape:
                    values = self.value_function(state)
                    new_policy_probs = self.call(state)
                    old_policy_probs = tf.stop_gradient(old_policy_probs)
                    new_policy_log = -1 * tf.math.log(tf.gather_nd(new_policy_probs, list(zip(np.arange(len(action)), action))))
                    old_policy_log = -1 * tf.math.log(tf.gather_nd(old_policy_probs, list(zip(np.arange(len(action)), action))))
                    ratio = tf.math.exp(tf.math.subtract(tf.math.log(new_policy_probs + 1e-10), tf.math.log(old_policy_probs + 1e-10)))
                    #ratio = tf.math.exp(old_policy_log - new_policy_log)
                    p1 = ratio * advantage
                    p2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                    actor_loss = -tf.reduce_mean(tf.minimum(p1, p2))
                    critic_loss = tf.reduce_mean(tf.math.squared_difference(reward, values))
                    e = -tf.reduce_sum(new_policy_probs * tf.math.log(tf.clip_by_value(new_policy_probs, self.non_zero, 1)), axis=1)
                    #e = -tf.reduce_sum(new_policy_probs * tf.math.log(new_policy_probs), axis=1)
                    entropy = tf.reduce_mean(e, axis=0)
                    #entropy = tf.reduce_mean(-(new_policy_probs * tf.math.log(new_policy_probs + 1e-10)))
                    loss = self.critic_discount * critic_loss + self.actor_discount * actor_loss - self.entropy_beta * entropy
                    #loss = - (actor_loss - self.constant_2 * critic_loss + self.constant_1 * entropy)
                gradients = tape.gradient(target = loss, sources = self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                episode_loss.append(loss)
        return tf.reduce_mean(episode_loss)


    def ppo_loss(self, states, actions, rewards, actions_probs, returns, advantages, log_probs, actions_onehot):
        episode_loss = []
        for _ in range(self.ppo_epochs):
            with tf.GradientTape() as tape:
                values = self.value_function(states)
                old_policy_probs = actions_probs
                new_policy_probs = self.call(states)
                old_policy_probs = tf.stop_gradient(old_policy_probs)
                new_policy_log = -1 * tf.math.log(tf.gather_nd(new_policy_probs, list(zip(np.arange(len(actions)), actions))))
                old_policy_log = -1 * tf.math.log(tf.gather_nd(old_policy_probs, list(zip(np.arange(len(actions)), actions))))
                ratio = tf.math.exp(tf.math.subtract(tf.math.log(new_policy_probs + 1e-10), tf.math.log(old_policy_probs + 1e-10)))
                #ratio = tf.math.exp(old_policy_log - new_policy_log)
                p1 = ratio * advantages
                p2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
                actor_loss = -tf.reduce_mean(tf.minimum(p1, p2))
                critic_loss = tf.reduce_mean(tf.math.squared_difference(rewards, values))
                e = -tf.reduce_sum(new_policy_probs * tf.math.log(tf.clip_by_value(new_policy_probs, self.non_zero, 1)), axis=1)
                #e = -tf.reduce_sum(new_policy_probs * tf.math.log(new_policy_probs), axis=1)
                entropy = tf.reduce_mean(e, axis=0)
                #entropy = tf.reduce_mean(-(new_policy_probs * tf.math.log(new_policy_probs + 1e-10)))
                loss = self.critic_discount * critic_loss + self.actor_discount * actor_loss - self.entropy_beta * entropy
                #loss = - (actor_loss - self.constant_2 * critic_loss + self.constant_1 * entropy)
            gradients = tape.gradient(target = loss, sources = self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            episode_loss.append(loss)
            
        return tf.reduce_mean(episode_loss)

#        values = self.value_function(states)
#        newpolicy_probs = self.call(states)
#        ratio = tf.math.exp(tf.math.subtract(tf.math.log(newpolicy_probs + 1e-10), tf.math.log(actions_probs + 1e-10)))
#        p1 = ratio * advantages
#        p2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
#        actor_loss = -tf.reduce_mean(tf.minimum(p1, p2))
#        critic_loss = tf.reduce_mean(tf.math.squared_difference(rewards, values))
#        #e = -tf.reduce_sum(newpolicy_probs * tf.math.log(tf.clip_by_value(newpolicy_probs, self.non_zero, 1)), axis=1)
#        e = -tf.reduce_sum(newpolicy_probs * tf.math.log(newpolicy_probs), axis=1)
#        entropy = tf.reduce_mean(e, axis=0)
#        #entropy = tf.reduce_mean(-(newpolicy_probs * tf.math.log(newpolicy_probs + 1e-10)))
#        loss = self.critic_discount * critic_loss + self.actor_discount * actor_loss - self.entropy_beta * entropy
#        #loss = - (actor_loss - self.constant_2 * critic_loss + self.constant_1 * entropy)
#
#        return loss