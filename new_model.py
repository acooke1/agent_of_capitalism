import tensorflow as tf
import numpy as np
import game_level as gl
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size

        # Define actor network parameters, critic network parameters, and optimizer
        self.hidden_size = 64
        self.critic_hidden_size = 32
        self.input_layer = tf.keras.layers.Input(shape=(state_size,))

        self.act_dense_layer1 = tf.keras.layers.Dense(self.hidden_size, use_bias=True) 
        self.act_dense_layer2 = tf.keras.layers.Dense(self.num_actions, use_bias=True)

        self.critic_dense_layer1 = tf.keras.layers.Dense(self.critic_hidden_size, use_bias=True) 
        self.critic_dense_layer2 = tf.keras.layers.Dense(1, use_bias=True) 

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # TODO tweak this!

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        # TODO: implement this!
        states = np.asarray(states)
        flattened_states = np.asarray(list(map(lambda x: x.flatten(), states)))
        dense_layer1 = self.act_dense_layer1(flattened_states)
        dense_layer1 = tf.nn.relu(dense_layer1)
        dense_layer2 = self.act_dense_layer2(dense_layer1)
        probabilites = tf.nn.softmax(dense_layer2)

        return probabilites

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode.
        :return: A [episode_length] matrix representing the value of each state.
        """
        # TODO: implement this :D
        states = np.asarray(states)
        flattened_states = np.asarray(list(map(lambda x: x.flatten(), states)))
        dense_layer1 = self.critic_dense_layer1(flattened_states)
        dense_layer1 = tf.nn.relu(dense_layer1)
        values = self.critic_dense_layer2(dense_layer1)

        return values

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the lecture slides referenced in the handout to see how this is done.

        Remember that the loss is similar to the loss as in part 1, with a few specific changes.

        1) In your actor loss, instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage. 
        See handout/slides for definition of advantage.
        
        2) In your actor loss, you must use tf.stop_gradient on the advantage to stop the loss calculated on the actor network 
        from propagating back to the critic network.
        
        3) See handout/slides for how to calculate the loss for your critic network.

        :param states: A batch of states of shape (episode_length, state_size)
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this :)
        # Hint: use tf.gather_nd (https://www.tensorflow.org/api_docs/python/tf/gather_nd) to get the probabilities of the actions taken by the model
        a_prob = self.call(states)
        indices = tf.range(0, tf.shape(a_prob)[0]) * self.num_actions + actions
        P_a = tf.gather(tf.reshape(a_prob, [-1]), indices)
        advantage = discounted_rewards - tf.squeeze(self.value_function(states))
        actor_loss = -tf.reduce_sum(tf.math.log(P_a) * tf.stop_gradient(advantage))
        critic_loss = tf.reduce_sum(tf.square(advantage))
        #import ipdb; ipdb.set_trace()

        return actor_loss+critic_loss

