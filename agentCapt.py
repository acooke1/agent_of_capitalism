import os
import sys
from pylab import *
import numpy as np
import tensorflow as tf
from model import Reinforce
import game_level as gl


def visualize_data(total_rewards):
    """
    HELPER - do not edit.
    Takes in array of rewards from each episode, visualizes reward over episodes

    :param total_rewards: List of rewards from all episodes
    """

    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()


def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each timestep in an episode, and
    returns a list of the discounted rewards for each timestep.
    Refer to the slides to see how this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """
    # TODO: Compute discounted rewards
    discounted_rewards = np.zeros(len(rewards))
    next_discounted_reward = 0.
    for i in range(len(rewards)-1, -1, -1):
        reward = rewards[i]
        next_discounted_reward = reward + (discount_factor * next_discounted_reward)
        discounted_rewards[i] = next_discounted_reward
    return discounted_rewards
        


def generate_trajectory(env, model):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    states = []
    actions = []
    rewards = []
    state = env.level_map
    done = False

    while not done:
        # TODO:
        # print('state shape', tf.expand_dims(state, axis = 0).shape)
        probs = model.call(tf.expand_dims(state, axis = 0))
        # model.num actions should always be two?
        #assert(model.num_actions == 2)
        # action is not a single integer here I don't think
        
        probs = tf.cast(probs, tf.float64)
        #print('probabilities: ', probs)
        #print('virgin probs', probs)
        action = np.random.choice([0, 1, 2, 3], 1, True, p=probs[0]/tf.reduce_sum(probs[0]))[0]

        #print('normalized probabilities', probs[0]/tf.reduce_sum(probs))
        #print('action', action)
        # 1) use model to generate probability distribution over next actions
        # 2) sample from this distribution to pick the next action

        states.append(state)
        actions.append(action)
        state, rwd, done = env.step(action)
        rewards.append(rwd)
        #print('reward', rwd)
        

    return states, actions, rewards


def train(env, model):
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one
    episode (lists of states, action_probs, and rewards seen/taken in the episode), and
    then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode

    :param env: The openai gym environment
    :param model: The model
    :returns: The total reward for the episode
    """

    # TODO:
    # 1) Use generate trajectory to run an episode and get states, actions, and rewards.
    
    with tf.GradientTape() as tape:
        states, actions, rewards = generate_trajectory(env, model)
        discounted_rewards = discount(rewards)
        episode_loss = model.loss(np.asarray(states), actions, discounted_rewards)
    gradients = tape.gradient(target = episode_loss, sources = model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # 2) Compute discounted rewards.
    # 3) Compute the loss from the model and run backpropagation on the model.
    #print(rewards)
    return tf.reduce_sum(rewards)


def main():

    env = gl.GameLevel(0)
    state_size = env.state_size
    num_actions = 4

    # Initialize model
    
    model = Reinforce(state_size, num_actions)

    # TODO: 
    rewards = []
    for i in range(250):
        episode_rewards = train(env, model)
        print('episode rewards: ', episode_rewards)
        rewards.append(np.sum(episode_rewards))
        # print('total episode rewards', episode_rewards)
    print(np.mean(np.asarray(rewards[200:])))
    visualize_data(rewards)
    # 1) Train your model for 650 episodes, passing in the environment and the agent. 
    # 2) Append the total reward of the episode into a list keeping track of all of the rewards. 
    # 3) After training, print the average of the last 50 rewards you've collected.

    # TODO: Visualize your rewards.


if __name__ == '__main__':
    main()
