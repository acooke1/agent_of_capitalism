import os
import sys
from pylab import *
import numpy as np
import tensorflow as tf
from model import Reinforce
from new_model import ReinforceWithBaseline
from ppo_model2 import PPOModel
import matplotlib.pyplot as plt
from IPython.display import clear_output
import game_level as gl
import glob

def plot_(frames, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frames, rewards[-1]))
    plt.plot(rewards)
    plt.show()

def get_gaes(values, masks, rewards, lmbda=0.95, gamma=0.99):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        #print(rewards[i], " ", gamma, " ", values[i+1], " ", masks[i], " ", values[i])
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, adv
    #return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

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
    :param print_map: A boolean corresponding to whether or not to print the game level after taking the step
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    actions_probs = []
    log_probs = []
    values = []
    states = []
    actions = []
    rewards = []
    masks = []
    actions_onehot = []
    env.reset_level()
    state = env.reset()
    done = False

    while not done:

        probs = model.call(tf.expand_dims(state, axis = 0))     
        probs = tf.cast(probs, tf.float64)

        q_value = np.array(model.value_function(tf.expand_dims(state, axis = 0)))

        action = np.random.choice([0, 1, 2, 3], 1, True, p=probs[0]/tf.reduce_sum(probs[0]))[0]
        
        action_onehot = np.zeros(4)
        action_onehot[action] = 1

        states.append(state)
        actions.append(action)
        actions_onehot.append(action_onehot)
        state, rwd, done = env.step(action)
        mask = not done
        masks.append(mask)
        rewards.append(rwd)
        values.append(q_value)
        actions_probs.append(probs)
        log_probs.append(tf.math.log(probs + 1e-10))

        if print_map:
            env.print_map()
            int_to_action = ["LEFT", "UP", "RIGHT", "DOWN", "ATTACK LEFT", "ATTACK UP", "ATTACK RIGHT", "ATTACK DOWN"]
            print("Action probabilities: ", probs)
            print("Action taken: " + int_to_action[action])
            print("Reward: " + str(rwd))

    q_value = np.array(model.value_function(tf.expand_dims(state, axis = 0)))
    values.append(q_value)
        
    return states, actions, rewards, values, actions_probs, actions_onehot, masks, log_probs


def train(env, model):
    """
    This function trains the model for one episode.

    :param env: The GameLevel
    :param model: The model
    :returns: The total reward for the episode
    """

    # Uses generate trajectory to run an episode and get states, actions, and rewards.
    states, actions, rewards, values, actions_probs, actions_onehot, masks, log_probs = generate_trajectory(env, model)

    #discounted_rewards = np.reshape(discount(rewards), (-1, 1))
    returns, advantages = get_gaes(values, masks, rewards)

    returns   = tf.stop_gradient(tf.concat(returns, axis=0)).numpy()
    log_probs = tf.stop_gradient(tf.concat(log_probs, axis=0))
    actions_probs = tf.stop_gradient(tf.concat(actions_probs, axis=0)).numpy()
    actions_onehot = tf.stack(actions_onehot).numpy()
    actions   = tf.stack(actions, axis=0).numpy()

    with tf.GradientTape() as tape:
        advantages = np.reshape(advantages, (-1, 1))
        rewards = np.reshape(rewards, (-1, 1))

        episode_loss = model.ppo_loss(np.asarray(states), actions, rewards, actions_probs, returns, advantages, log_probs, actions_onehot)
    #gradients = tape.gradient(target = episode_loss, sources = model.trainable_variables)
    #model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return tf.reduce_sum(rewards), len(rewards), episode_loss

def test_env(env, model, print_map=False, use_random_map=True):
    env.reset_level()
    state = env.reset()

    done = False
    total_reward = 0
    steps_needed = 0
    while not done:

        probs = model.call(tf.expand_dims(state, axis = 0))     
        probs = tf.cast(probs, tf.float64)

        action = np.random.choice([0, 1, 2, 3], 1, True, p=probs[0]/tf.reduce_sum(probs[0]))[0]

        state, rwd, done = env.step(action)

        total_reward += rwd
        steps_needed += 1

        if print_map:
            env.print_map()
            int_to_action = ["LEFT", "UP", "RIGHT", "DOWN", "ATTACK LEFT", "ATTACK UP", "ATTACK RIGHT", "ATTACK DOWN"]
            print("Action probabilities: ", probs)
            print("Action taken: " + int_to_action[action])
            print("Reward: " + str(rwd))

    return total_reward, steps_needed

def main():
    # PARAMETERS FOR THIS TRAINING RUN
    game_level = 1
    use_submap = False
    use_enemy = True
    allow_attacking = False
    epochs = 300

    # PARAMETERS FOR RANDOM MAP GENERATION
    use_random_maps = False
    test_random_maps = False
    side_length = 8 # Generally, values between 8 and 16 are good
    wall_prop = 0.3 # This is the fraction of empty spaces that become walls. Generally, values between 0.25 and 0.35 are good
    num_coins = 7
    starting_pos = [1,1] # Setting this to [1,1] is standard (top-left corner), but if you wanted, you could set it to [4,5], or other starting positions

    # Initialize the game level
    env = gl.GameLevel(game_level, use_enemy, use_submap, use_random_maps, side_length, wall_prop, num_coins, starting_pos)
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
    model = PPOModel(state_size, num_actions)

    rewards = []
    test_rewards = []
    frames_ = []
    steps_required = []
    
    # Train for num_epochs epochs
    for i in range(epochs):
        episode_rewards, episode_length, episode_loss = train(env, model)
        if (i % 10 == 0):
            test_reward = 0
            avg_steps_needed = 0
            cntr = 0
            for _ in range(20):
                test_env_ = env
                if (test_random_maps):
                    test_env_ = gl.GameLevel(game_level, use_enemy, use_submap, test_random_maps, side_length, wall_prop, num_coins, starting_pos)
                t, a = test_env(test_env_, model, print_map=False)
                test_reward += t
                avg_steps_needed += a
                cntr += 1
            avg_steps_needed = avg_steps_needed/cntr
            test_reward = test_reward/cntr
            test_rewards.append(test_reward)
            steps_required.append(avg_steps_needed)
            frames_.append(i)
            print('Episode: ' + str(i) +', episode length: ', episode_length, ', episode rewards: ', episode_rewards.numpy(), ', episode loss: ', episode_loss)
        rewards.append(np.sum(episode_rewards))
        # print('total episode rewards', episode_rewards)
    
    # Run the model once, printing its movements this time
    generate_trajectory(env, model, print_map=True)
    #print(np.mean(np.asarray(rewards[50:])))
    visualize_data(rewards)
    plot_(frames_, test_rewards)
    plot_(frames_, steps_required)


if __name__ == '__main__':
    main()
