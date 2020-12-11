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
        


def generate_trajectory(env, model, print_map=False, calc_value=False):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The GameLevel
    :param model: The model used to generate the actions
    :param print_map: A boolean corresponding to whether or not to print the game level after taking the step
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    states = []
    actions = []
    rewards = []
    old_probs = []
    masks = []
    state = env.reset()
    done = False

    # num_wall_collisions = 0

    while not done:
        # print('state shape', tf.expand_dims(state, axis = 0).shape)

        # Calls the model to generate probability distribution for next possible actions
        probs = model.call(tf.expand_dims(state, axis = 0))        
        probs = tf.cast(probs, tf.float64)
        old_probs.append(probs)

        # Randomly samples from the distribution to determine the next action
        action = np.random.choice([0, 1, 2, 3], 1, True, p=probs[0]/tf.reduce_sum(probs[0]))[0]

        # Stores the chosen state, action, and reward for the step, and calls the GameLevel to get the next state
        states.append(state)
        actions.append(action)
        state, rwd, done = env.step(action)
        mask = not done
        masks.append(mask)
        rewards.append(rwd)

        if print_map:
            env.print_map()
            int_to_action = ["LEFT", "UP", "RIGHT", "DOWN", "ATTACK LEFT", "ATTACK UP", "ATTACK RIGHT", "ATTACK DOWN"]
            print("Action probabilities: ", probs)
            print("Action taken: " + int_to_action[action])
            print("Reward: " + str(rwd))
            print()

    old_probs = tf.stop_gradient(tf.concat(old_probs, axis=0)).numpy()

    return states, actions, rewards, old_probs


def train(env, model, model_type):
    """
    This function trains the model for one episode.

    :param env: The GameLevel
    :param model: The model
    :returns: The total reward for the episode
    """

    # Uses generate trajectory to run an episode and get states, actions, and rewards.
    states, actions, rewards, old_probs = generate_trajectory(env, model, False, not(model_type == "REINFORCE"))
    with tf.GradientTape() as tape:
        discounted_rewards = discount(rewards)
        # Computes loss from the model and runs backpropagation
        if (model_type == "PPO"):
            episode_loss = model.loss(np.asarray(states), actions, discounted_rewards, old_probs)
        else:
            episode_loss = model.loss(np.asarray(states), actions, discounted_rewards)
    gradients = tape.gradient(target = episode_loss, sources = model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    #print(rewards)
    return tf.reduce_sum(rewards), len(rewards), old_probs, episode_loss


def main():
    # PARAMETERS FOR THIS TRAINING RUN
    game_level = 2
    use_submap = True
    use_enemy = False
    allow_attacking = False
    num_epochs = 5000

    # PARAMETERS FOR RANDOM MAP GENERATION
    use_random_maps = True # NOTE: when use_random_maps is True, the enemy may not necessarily work unless use_random_starts is also True
    side_length = 8 # Generally, values between 8 and 16 are good
    wall_prop = 0.3 # This is the fraction of empty spaces that become walls. Generally, values between 0.25 and 0.35 are good
    num_coins = 8
    starting_pos = [1,1] # Setting this to [1,1] is standard (top-left corner), but if you wanted, you could set it to [4,5], or other starting positions
    use_random_starts = True

    # Initialize the game level
    env = gl.GameLevel(game_level, use_enemy, use_submap, use_random_maps, side_length, wall_prop, num_coins, starting_pos, use_random_starts)
    # NOTE: all parameters after game_level are entirely optional, just passed here so that the setting options above work properly

    # PARAMETERS FOR THE MODEL
    if use_submap:
        state_size = env.submap_dims**2
    else:
        state_size = env.level_area

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
    # Train for num_epochs epochs
    for i in range(num_epochs):
        episode_rewards, episode_length, old_probs, episode_loss = train(env, model, sys.argv[1])
        print('Episode: ' + str(i) +', episode length: ', episode_length, ', episode rewards: ', episode_rewards.numpy(), ', episode loss: ', episode_loss.numpy())
        rewards.append(np.sum(episode_rewards))
        # print('total episode rewards', episode_rewards)
    
    # Run the model once, printing its movements this time
    generate_trajectory(env, model, print_map=True)
    #print(np.mean(np.asarray(rewards[50:])))
    visualize_data(rewards)


if __name__ == '__main__':
    main()
