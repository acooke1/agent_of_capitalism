import tensorflow as tf
import numpy as np
import game_level as gl

class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions
        self.state_size = state_size

        # Define actor network parameters, critic network parameters, and optimizer
        self.optimizer = tf.keras.optimizers.Adam(.005)
        self.dense1 = tf.keras.layers.Dense(self.state_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.state_size, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_actions, activation='softmax')
        pass

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
        output = self.dense1(states)
        output = self.dense2(output)
        output = self.dense3(output)
        
        return output

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
        
        probs = self.call(states)
        prob_for_actions = tf.gather_nd(probs, list(zip(np.arange(probs.shape[0]), actions)))
        log_prob_actions = tf.math.log(prob_for_actions)
        #print('shape of discounted rewards', discounted_rewards.shape)
        #print('episode length', states.shape[0])
        loss_actor = -tf.reduce_sum(log_prob_actions * discounted_rewards)
        
        #tf.stop_gradient(self.value_function(states))
        
        return loss_actor


