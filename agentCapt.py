import os
import sys
from pylab import *
import numpy as np
import tensorflow as tf
from model import Reinforce
from new_model import ReinforceWithBaseline
from ppo_model import PPOModel
import game_level as gl
import glob


def visualize_data(total_rewards):
    """
    HELPER FUNCTION
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

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """
    discounted_rewards = np.zeros(len(rewards))
    next_discounted_reward = 0.
    for i in range(len(rewards)-1, -1, -1):
        reward = rewards[i]
        next_discounted_reward = reward + (discount_factor * next_discounted_reward)
        discounted_rewards[i] = next_discounted_reward
    return discounted_rewards
        


def generate_trajectory(env, model, print_map=False):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The GameLevel
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    states = []
    actions = []
    rewards = []
    state = env.reset()
    done = False

    while not done:
        # print('state shape', tf.expand_dims(state, axis = 0).shape)

        # Calls the model to generate probability distribution for next possible actions
        probs = model.call(tf.expand_dims(state, axis = 0))        
        probs = tf.cast(probs, tf.float64)

        # Randomly samples from the distribution to determine the next action
        action = np.random.choice([0, 1, 2, 3], 1, True, p=probs[0]/tf.reduce_sum(probs[0]))[0]

        # Stores the chosen state, action, and reward for the step, and calls the GameLevel to get the next state
        states.append(state)
        actions.append(action)
        state, rwd, done = env.step(action)
        rewards.append(rwd)

        if print_map:
            env.print_map()
            int_to_action = ["LEFT", "UP", "RIGHT", "DOWN", "ATTACK LEFT", "ATTACK UP", "ATTACK RIGHT", "ATTACK DOWN"]
            print("Action probabilities: ", probs)
            print("Action taken: " + int_to_action[action])
            print("Reward: " + str(rwd))
        
    return states, actions, rewards


def train(env, model, previous_actions, old_probs, model_type):
    """
    This function trains the model for one episode.

    :param env: The GameLevel
    :param model: The model
    :returns: The total reward for the episode
    """

    # Uses generate trajectory to run an episode and get states, actions, and rewards.
    with tf.GradientTape() as tape:
        states, actions, rewards = generate_trajectory(env, model)
        discounted_rewards = discount(rewards)
        # Computes loss from the model and runs backpropagation
        if (model_type == "PPO"):
            episode_loss, old_probs = model.loss(np.asarray(states), actions, rewards, previous_actions, old_probs)
        else:
            episode_loss, old_probs = model.loss(np.asarray(states), actions, discounted_rewards)
    gradients = tape.gradient(target = episode_loss, sources = model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    #print(rewards)
    return tf.reduce_sum(rewards), len(rewards), old_probs, actions, episode_loss


def main():
    # PARAMETERS FOR THIS TRAINING RUN
    game_level = 0
    use_enemy = False
    allow_attacking = False
    num_epochs = 450

    # Initialize the game level
    env = gl.GameLevel(game_level, use_enemy)

    # PARAMETERS FOR THE MODEL
    state_size = env.state_size
    if allow_attacking:
        num_actions = 8
    else:
        num_actions = 4

    # Initialize model
    if sys.argv[1] == "REINFORCE":
        model = Reinforce(state_size, num_actions) 
    elif sys.argv[1] == "REINFORCE_BASELINE":
        model = ReinforceWithBaseline(state_size, num_actions)
    elif sys.argv[1] == "PPO":
        model = PPOModel(state_size, num_actions)
    else:
        print("INCORRECT CALL. CALL SHOULD BE OF FORMAT: python assignment.py REINFORCE/REINFORCE_BASELINE/PPO")
        exit()
    # model = ReinforceWithBaseline(state_size, num_actions)

    rewards = []
    previous_actions = []
    old_probs = tf.Variable(np.zeros((1,model.num_actions)) + 0.25, dtype=tf.float32)
    # Train for num_epochs epochs
    for i in range(num_epochs):
        episode_rewards, episode_length, old_probs, previous_actions, episode_loss = train(env, model, previous_actions, old_probs, sys.argv[1])
        print('Episode: ' + str(i) +', episode length: ', episode_length, ', episode rewards: ', episode_rewards.numpy(), ', episode loss: ', episode_loss.numpy())
        rewards.append(np.sum(episode_rewards))
        # print('total episode rewards', episode_rewards)
    
    # Run the model once, printing its movements this time
    generate_trajectory(env, model, False)
    #print(np.mean(np.asarray(rewards[50:])))
    visualize_data(rewards)


if __name__ == '__main__':
    main()
