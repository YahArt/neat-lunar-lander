import os
import pickle
import neat
import gym
import numpy as np
import time

#load the winner
with open('winnerIterations/winner-feedforward', 'rb') as f:
	c = pickle.load(f)


#Load the config file, which is assumed to live in the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config/config-feedforward.txt')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
net = neat.nn.FeedForwardNetwork.create(c, config)

env = gym.make('LunarLander-v2')
observation = env.reset()



done = False

while not done:
	time.sleep(0.02)
	action = np.argmax(net.activate(observation))
	observation, reward, done, info = env.step(action)
	env.render()