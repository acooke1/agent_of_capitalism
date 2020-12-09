import os
import gym
import numpy as np
import tensorflow as tf
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001
gamma = 0.99
lmbda = 0.95
num_actions = 4

def actor_model(input_dims, output_dims):
    state_input = tf.keras.layers.Input(shape=input_dims)
    oldpolicy_probs = tf.keras.layers.Input(shape=(1, output_dims,))
    advantages = tf.keras.layers.Input(shape=(1, 1,))
    rewards = tf.keras.layers.Input(shape=(1, 1,))
    values = tf.keras.layers.Input(shape=(1, 1,))

    # Classification block
    x = tf.keras.layers.Dense(50, activation='relu', name='fc1')(state_input)
    x = tf.keras.layers.Dense(25, activation='relu', name='fc2')(x)
    out_actions = tf.keras.layers.Dense(num_actions, activation='softmax', name='predictions')(x)

    model = tf.keras.models.Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[out_actions])
    model.compile(optimizer=tf.optimizers.Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
    # model.summary()
    return model

def critic_model(input_dims):
    state_input = tf.keras.layers.Input(shape=input_dims)

    # Classification block
    x = tf.keras.layers.Dense(50, activation='relu', name='fc1')(state_input)
    x = tf.keras.layers.Dense(25, activation='relu', name='fc2')(x)
    out_actions = tf.keras.layers.Dense(1, activation='tanh')(x)

    model = tf.keras.models.Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=tf.optimizers.Adam(lr=1e-4), loss='mse')
    # model.summary()
    return model



def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = tf.keras.backend.exp(tf.keras.backend.log(newpolicy_probs + 1e-10) - tf.keras.backend.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = tf.keras.backend.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -tf.keras.backend.mean(tf.keras.backend.minimum(p1, p2))
        critic_loss = tf.keras.backend.mean(tf.keras.backend.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * tf.keras.backend.mean(
            -(newpolicy_probs * tf.keras.backend.log(newpolicy_probs + 1e-10)))
        return total_loss

    return loss


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

        self.mini_batch_size = 5
        self.ppo_epochs = 5

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

    def loss(self, states, actions, rewards, actions_probs, advantages):

        values = self.value_function(states)
        newpolicy_probs = self.call(states)
        actions_probs = tf.cast(tf.concat(actions_probs, axis=0), dtype=tf.float32)
        #print(newpolicy_probs)
        #print(actions_probs)
        ratio = tf.math.exp(tf.math.log(newpolicy_probs + 1e-10) - tf.math.log(actions_probs + 1e-10))
        p1 = ratio * advantages
        p2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -tf.reduce_mean(tf.minimum(p1, p2))
        critic_loss = tf.reduce_mean(tf.math.squared_difference(rewards, values))
        e = -tf.reduce_sum(newpolicy_probs * tf.math.log(tf.clip_by_value(newpolicy_probs, self.non_zero, 1)), axis=1)
        entropy = tf.reduce_mean(e, axis=0)
        #entropy = tf.reduce_mean(-(newpolicy_probs * tf.math.log(newpolicy_probs + 1e-10)))
        loss = self.critic_discount * critic_loss + self.actor_discount * actor_loss - self.entropy_beta * entropy
        #loss = - (actor_loss - self.constant_2 * critic_loss + self.constant_1 * entropy)
        return loss

    def ppo_iter(self, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // self.mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

    def ppo_loss2(self, states, actions, log_probs, returns, advantages):
        for state, action, old_log_probs, return_, advantage in self.ppo_iter(states, actions, log_probs, returns, advantages):
            dist = model.call(states)
            value = model.value_function(states)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage

            actor_loss  = - tf.minimum(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            return loss
