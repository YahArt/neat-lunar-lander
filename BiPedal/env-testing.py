#Import Bereich
import gym #für saubere installation use https://anaconda.org/conda-forge/gym
#import Box2D
#Neat und gym

#enviroment für gym erstellen
#something like:
env = gym.make('LunarLander-v2')

print("Hello")

observation = env.reset()

print(observation) #8 Values: pos X, pos Y, x velocity, y velocity, lander angle, angular velocity, let contact point, right contact point
print(env.action_space) #Discrete(4) --> hat 4 Action Möglichkeiten: do noting [0], left engine [1], main engine [2], right engine [3]?

done = False

while not done:
    observation, reward, done, info = env.step(env.action_space.sample()) #action_space.sample lässt den moonlander einfach zufällig Aktionen ausführen
    print(env.action_space.sample())

    env.render()

print(observation)
