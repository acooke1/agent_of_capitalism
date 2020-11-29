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
        self.E = 0

    
    def generate_trajectories(self, env):
        # num_visits tracks the number of times each (s,a) pair is seen throughout the episodes. Used 
        # to generate T and R after running n_games episodes
        total_rwd = 0

        states = []
        actions = []
        rewards = []
        state = env.level_map
        state = list(np.asarray(state).flatten()).index(3)
        done = False
        #print(env.level_map)
        #print(self.E)
        while not done:
            # print('state shape', tf.expand_dims(state, axis = 0).shape)
            # Calls the model to generate probability distribution for next possible actions
            if np.random.rand(1) < .6:
                action = np.random.randint(0,4)
            else:
                action = np.argmax(self.Q[state])
            #print('probabilities: ', probs)
            #print('virgin probs', probs)

            # Randomly samples from the distribution to determine the next action
            # action = np.random.choice([0, 1, 2, 3], 1, True, p=probs[0]/tf.reduce_sum(probs[0]))[0]

            #print('normalized probabilities', probs[0]/tf.reduce_sum(probs))
            #print('action', action)

            # Stores the chosen state, action, and reward for the step, and calls the GameLevel to get the next state
            states.append(state)
            actions.append(action)
            state, rwd, done = env.step(action)
            state = list(np.asarray(state).flatten()).index(3)
            total_rwd += rwd
            rewards.append(rwd)
            self.Q[states[-1]][action] = (1-self.alpha) * self.Q[states[-1]][action] + self.alpha * (rwd + self.gamma * self.V[state])
            self.V[states[-1]] = np.max(self.Q[states[-1]])
    
            #print('reward', rwd)
        
        # print('states: ', states)
        # print('rewards: ', rewards)
            
        return np.sum(rewards)

def main():
    E = 1000
    env = gl.GameLevel(0, False)
    map_length = len(env.level_map)
    model = QLearning((map_length, map_length), 1000, .99, .2, 4)
    total_rwds = []
    for i in range(15000):
        model.E = model.epsilon/(model.epsilon + i)
        total_rwd_epoch = model.generate_trajectories(env)
        total_rwds.append(total_rwd_epoch)
        print('epoch rewards: ', total_rwd_epoch)
    print('avg rewards over last 50:', np.mean(total_rwds[7000:]))
    ac.visualize_data(total_rwds)

if __name__ == "__main__":
    main()

