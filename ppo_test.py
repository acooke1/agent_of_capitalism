import math
import random

import gym
import numpy as np
import game_level as gl
import tensorflow as tf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from IPython.display import clear_output
import matplotlib.pyplot as plt
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

game_level = 0
use_enemy = False
use_submap = False
use_random_maps = False
side_length = 8 
wall_prop = 0.3 
num_coins = 8
starting_pos = [1,1] 
use_random_starts = True
non_zero = 1e-7

env = gl.GameLevel(game_level, use_enemy, use_submap, use_random_maps, side_length, wall_prop, num_coins, starting_pos)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
        

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax()
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().squeeze().expand_as(mu)
        #std   = self.log_std.squeeze().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
def test_env(vis=False):
    env.reset_level()
    state = env.reset()

    done = False
    total_reward = 0
    steps_needed = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = dist.sample()[0]
        norm = action.cpu().numpy()
        norm = [(float(i)-min(norm))/(max(norm)-min(norm)) for i in norm]
        s = sum(norm)
        norm = norm/s
        a = np.random.choice([0, 1, 2, 3], 1, True, p=norm)[0]
        next_state, reward, done = env.step(a)
        state = next_state
        
        total_reward += reward
        steps_needed += 1

    return total_reward, steps_needed

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        print(actions)
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        #print(states.size())
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        
        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.1):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
        
            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = value_coef * critic_loss + actor_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

num_inputs  = env.level_area
num_outputs = 4

#Hyper params:
hidden_size      = 64
lr               = 5e-5
num_steps        = 128
mini_batch_size  = 20
ppo_epochs       = 10
threshold_reward = 10
value_coef = 0.34
entropy_coef = 0.02

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

max_frames = 30000
frame_idx  = 0
test_rewards = []
frames = []
required_steps = []

early_stop = False

while frame_idx < max_frames and not early_stop:

    env.reset_level()
    state = env.reset()

    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    entropy = 0
    done = False

    for _ in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        dist, value = model(torch.unsqueeze(state, dim=0))

        action = dist.sample()[0]
        norm = action.cpu().numpy()
        norm = [(float(i)-min(norm))/(max(norm)-min(norm)) for i in norm]
        s = sum(norm)
        norm = norm/s
        #print(norm, " ", s)
        #print(action.cpu().numpy())
        a = np.random.choice([0, 1, 2, 3], 1, True, p=norm)[0]
        next_state, reward, done = env.step(a)

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor([float(1 - done)]).unsqueeze(1).to(device))
        
        states.append(state)
        actions.append(action)
        
        state = next_state
        frame_idx += 1
        
        if frame_idx % 500 == 0:
            test_reward = 0
            avg_steps_needed = 0
            cntr = 0
            for _ in range(10):
                t, a = test_env()
                test_reward += t
                avg_steps_needed += a
                cntr += 1
            test_reward = test_reward/cntr
            avg_steps_needed = avg_steps_needed/cntr
            test_rewards.append(test_reward)
            frames.append(frame_idx)
            required_steps.append(avg_steps_needed)
            print(frame_idx, test_reward, avg_steps_needed)
            if test_reward > threshold_reward: early_stop = True
        
        if (done):
            break
            

    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = compute_gae(next_value, rewards, masks, values)

    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.stack(states)
    actions   = torch.stack(actions)
    advantage = returns - values
    
    ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

plot(frame_idx, test_rewards)
plot(frame_idx, required_steps)