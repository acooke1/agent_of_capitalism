import numpy as np
import sys
import random as random
import tensorflow as tf
import game_level as gl
import agentCapt as ac

class QLearning(tf.keras.Model):
    def __init__(self, shape, epsilon, gamma, alpha, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(QLearning, self).__init__()
        self.num_actions = num_actions
        self.V = np.zeros(shape[0]**2)
        self.Q = np.zeros((shape[0]**2, num_actions)) 
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.E = epsilon

    
    def generate_trajectories(self, env, E, print_map=False):
        # num_visits tracks the number of times each (s,a) pair is seen throughout the episodes. Used 
        # to generate T and R after running n_games episodes
        total_rwd = 0

        states = []
        actions = []
        rewards = []
        state = env.reset()
        state = list(state).index(env.player_level_val)
        done = False
        coins_left = 0
        while not done:
            coins_left = env.num_coins_left
            # Calls the model to generate probability distribution for next possible actions
            if np.random.rand(1) < E:
                action = np.random.randint(0,4)
            else:
                action = np.argmax(self.Q[state])
            # Stores the chosen state, action, and reward for the step, and calls the GameLevel to get the next state
            states.append(state)
            actions.append(action)
            state, rwd, done = env.step(action)
            state = list(state).index(env.player_level_val)
            total_rwd += rwd
            rewards.append(rwd)
            self.Q[states[-1]][action] = (1-self.alpha) * self.Q[states[-1]][action] + self.alpha * (rwd + self.gamma * self.V[state])
            self.V[states[-1]] = np.max(self.Q[states[-1]])
            if print_map:
                env.print_map()
                int_to_action = ["LEFT", "UP", "RIGHT", "DOWN", "ATTACK LEFT", "ATTACK UP", "ATTACK RIGHT", "ATTACK DOWN"]
                print("Action taken: " + int_to_action[action])
                print("Reward: " + str(rwd))
                print()
    
            #print('reward', rwd)
        
        # print('states: ', states)
        # print('rewards: ', rewards)
        print('coins left: ', coins_left)
        printValue(self, env)
        return np.sum(rewards), coins_left, actions

def printValue(model, env):
    V = np.reshape(model.V, [env.side_length, env.side_length])
    for i in range(V.shape[0]):
        print(V[i])

def main():
    E = 1000
    env = gl.GameLevel(level =0, use_submap=False, use_random_maps=False, side_length = 8, use_random_starts=False, use_random_seed=True)
    # env = gl.GameLevel(level =0, use_submap= F)
    map_length = len(env.level_map)
    model = QLearning((map_length, map_length), 1000, .99, .1, 4)
    total_rwds = []
    E = .6
    recent_wins = 0
    wins = 0
    for i in range(5000):
        if i > 2000:
            E = .4
        if i > 3500:
            E = .2
        print('curr epoch: ', i+1)
        model.E = model.epsilon/(model.epsilon + i)
        if i == 0 or i == 4999:
            total_rwd_epoch, coins_left, actions = model.generate_trajectories(env, E, print_map = True)
        else: 
            total_rwd_epoch, coins_left, actions = model.generate_trajectories(env, E)
        if coins_left == 1:
            if i > 4500:
                recent_wins+=1
            wins+=1
        total_rwds.append(total_rwd_epoch)
        #print('epoch rewards: ', total_rwd_epoch)
    print('avg rewards over last 50:', np.mean(total_rwds[4500:]))
    print('number of wins: ', wins)
    print('number of recent wins: ', recent_wins)
    ac.visualize_data(total_rwds)

if __name__ == "__main__":
    main()

