import os
import sys
from pylab import *
import tensorflow as tf
import numpy as np
import game_level as gl
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class QModel(tf.keras.Model):
	def __init__(self, state_size, num_actions):
		
		super(QModel, self).__init__()
		self.num_actions = num_actions
		self.state_size = state_size

		self.advantage_hidden_size = 64
		self.value_hidden_size = 32
		self.learning_rate = 0.001 

		self.input_layer = tf.keras.layers.Input(shape=(state_size,))

		self.advantage_dense_layer1 = tf.keras.layers.Dense(self.advantage_hidden_size, use_bias=True)
		self.advantage_dense_layer2 = tf.keras.layers.Dense(self.num_actions, use_bias=True)

		self.value_dense_layer1 = tf.keras.layers.Dense(self.value_hidden_size, use_bias=True)
		self.value_dense_layer2 = tf.keras.layers.Dense(1, use_bias=True) 

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate) 


	def advantage(self, states):
		
		states = np.asarray(states)
		flattened_states = np.asarray(list(map(lambda x: x.flatten(), states)))
		dense_layer1 = self.advantage_dense_layer1(flattened_states)
		dense_layer1 = tf.nn.relu(dense_layer1)
		advantage = self.advantage_dense_layer2(dense_layer1)

		return advantage

	def value(self, states):
		
		states = np.asarray(states)
		flattened_states = np.asarray(list(map(lambda x: x.flatten(), states)))
		dense_layer1 = self.value_dense_layer1(flattened_states)
		dense_layer1 = tf.nn.relu(dense_layer1)
		values = self.value_dense_layer2(dense_layer1)

		return values

	def q_values(self, states):

		values = self.value(states)
		advantage = self.advantage(states)

		return values + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))


	def loss(self, q_values, target_qs):

		return tf.reduce_sum(tf.square(q_values - target_qs))

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


def generate_trajectory(env, model, print_map=False):
	"""
	Generates lists of states, actions, and rewards for one complete episode.

	:param env: The GameLevel
	:param model: The model used to generate the actions
	:param print_map: A boolean corresponding to whether or not to print the game level after taking the step
	:returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
	"""
	epsilon = 0.3
	gamma = 0.99
	states = []
	actions = []
	rewards = []
	target_qs = []
	state = env.reset()
	done = False
	replay_buffer = []
	max_replay_buffer_size = 100

	initialize_replay_buffer(replay_buffer, 50, env, model)

	while not done:
		# print('state shape', tf.expand_dims(state, axis = 0).shape)
		Qs  = model.q_values(state)
		# Calls the model to generate probability distribution for next possible actions
		if np.random.random() <= epsilon:
			action = np.random.choice(model.num_actions)
		else:
			action = np.argmax(Qs,1)[0]

		# Stores the chosen state, action, and reward for the step, and calls the GameLevel to get the next state
		states.append(state)
		actions.append(action)
		state, rwd, done = env.step(action)
		rewards.append(rwd)

		if len(replay_buffer) == max_replay_buffer_size:
			replay_buffer.pop(0)
		replay_buffer.append(np.array([action, rwd, done]))

		next_Qs = model.q_values(state)
		target_qs = Qs.numpy()
		target_qs[0][action] =  rwd + gamma*np.max(next_Qs)

		if print_map:
			env.print_map()
			int_to_action = ["LEFT", "UP", "RIGHT", "DOWN", "ATTACK LEFT", "ATTACK UP", "ATTACK RIGHT", "ATTACK DOWN"]
			#print("Action probabilities: ", Qs)
			print("Action taken: " + int_to_action[action])
			print("Reward: " + str(rwd))
		
	return states, actions, rewards, Qs, target_qs


def initialize_replay_buffer(replay_buffer, init_size, env, model):
	state = env.reset()
	for i in range(init_size):
		action = np.random.choice(model.num_actions)
		next_state, reward, done = env.step(action)
		replay_buffer.append(np.array([action, reward, done]))
		if done:
			state = env.reset()


def train(env, model, previous_actions, old_probs):
	"""
	This function trains the model for one episode.

	:param env: The GameLevel
	:param model: The model
	:returns: The total reward for the episode
	"""

	# Uses generate trajectory to run an episode and get states, actions, and rewards.
	with tf.GradientTape(persistent=True) as tape:
		gamma = 0.99
		states, actions, rewards, q_values, target_qs = generate_trajectory(env, model)
		loss = model.loss(q_values, target_qs)
	gradients = tape.gradient(target = loss, sources = model.trainable_variables)
	model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
	
	return tf.reduce_sum(rewards), len(rewards), old_probs, actions, loss

def main():
	# PARAMETERS FOR THIS TRAINING RUN
	game_level = 0
	use_submap = False
	use_enemy = False
	allow_attacking = False
	num_epochs = 1000

	# Initialize the game level
	env = gl.GameLevel(game_level, use_enemy, use_submap)

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
	model = QModel(state_size, num_actions) 

	rewards = []
	previous_actions = []
	old_probs = tf.Variable(np.zeros((1,model.num_actions)) + 0.25, dtype=tf.float32)
	# Train for num_epochs epochs
	for i in range(num_epochs):
		episode_rewards, episode_length, old_probs, previous_actions, episode_loss = train(env, model, previous_actions, old_probs)
		print('Episode: ' + str(i) +', episode length: ', episode_length, ', episode rewards: ', episode_rewards.numpy(), ', episode loss: ', episode_loss.numpy())
		rewards.append(np.sum(episode_rewards))
	
	# Run the model once, printing its movements this time
	generate_trajectory(env, model, print_map=True)
	#print(np.mean(np.asarray(rewards[50:])))
	visualize_data(rewards)


if __name__ == '__main__':
	main()
