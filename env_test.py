#Import Bereich
import gym #für saubere installation use https://anaconda.org/conda-forge/gym
#import Box2D
#Neat und gym

#enviroment für gym erstellen
#something like:
env = gym.make('LunarLander-v2')


observation = env.reset()

print(observation) #8 Values: pos X, pos Y, x velocity, y velocity, lander angle, angular velocity, let contact point, right contact point
print(env.action_space) #Discrete(4) --> hat 4 Action Möglichkeiten: do noting [0], left engine [1], main engine [2], right engine [3]?

done = False

while not done:
    observation, reward, done, info = env.step(env.action_space.sample()) #action_space.sample lässt den moonlander einfach zufällig Aktionen ausführen
    x_pos = observation[0] # Between -1 and 1
    angle = observation[4] # Between ?
    velocity = observation[5] #
    contact_leg_one = observation[6] == 1.0
    contact_leg_two = observation[7] == 1.0
    print("X Pos: ", x_pos)
    print("Contact Leg One: ", contact_leg_one)
    print("Contact Leg Two: ", contact_leg_two)
    print("Angle: ", angle)
    print("Velocity: ", velocity)
    env.render()
