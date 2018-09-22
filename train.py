'''
Solve OpenAI Gym's CartPole-V1 with Q-learning table
'''

import gym
import numpy as np
import math


'''
Training Class
'''
class CatrPoleSolver:
	def __init__(self):
		# Challenge is solved if we can maintain
		# the pole straight for 200 steps
		self.objective = 200

		# Load CartPole
		self.env = gym.make('CartPole-v1')
		
		 # learning rate
		self.learningRate = 0.1

		# exploration rate
		self.explorationRate = 0.3
		
		# Create a 6 * 12 * 2 feature space
		self.features = (1, 1, 6, 12, self.env.action_space.n)
		
		# Create Q Table
		self.QTable = np.zeros(self.features)

		# Start training
		self.run()


	def observe(self, observation):
		# Transform observation to discrete values
		upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
		lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
		ratios = [(observation[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(observation))]
		new_obs = [int(round((self.features[i] - 1) * ratios[i])) for i in [0, 1, 2, 3]]
		new_obs = [min(self.features[i] - 1, max(0, new_obs[i])) for i in [0, 1, 2, 3]]
		return tuple(new_obs)

	def choose_action(self, state):
		return self.env.action_space.sample() if (np.random.random() <= self.explorationRate) else np.argmax(self.QTable[state])

	def decay_rate(self, step, minimum):
		return max(minimum, min(1, 1.0 - math.log(step + 1)))

	def run(self):

		solved = False
		steps = 0
		game = 0
		while not solved:
			# env.reset()
			current_state = self.observe(self.env.reset())
			done = False;
			score = 0
			game += 1
			step = 0

			while not done:
				# Render game
				# self.env.render()

				# Choose action
				action = self.choose_action(current_state)
				observation, reward, done, info = self.env.step(action) # take a random action
				new_state = self.observe(observation)

				# learn
				self.QTable[current_state][action] += self.decay_rate(step, self.learningRate) * (reward + 1 * np.max(self.QTable[new_state]) - self.QTable[current_state][action])

				# Update state
				current_state = new_state
				score += 1
				step += 1
				steps += 1

				# Did we solve this challenge?
				if score == self.objective:
					done = True
					solved = True


		print '----'
		print 'Solved in {} steps'.format(steps)
		print '----'


if __name__ == "__main__":
	CatrPoleSolver()