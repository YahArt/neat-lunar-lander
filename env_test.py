#Import Bereich
import gym #für saubere installation use https://anaconda.org/conda-forge/gym
import numpy as np


def evaluate_fitness(pos_current, pos_old, vel_current, vel_old, angular_vel_current, angular_vel_old, has_landed):
    fitness = 0.0


    # −100 ∗ (dt − dt−1) − 100 ∗ (vt − vt−1) −100 ∗ (ωt − ωt−1) + hasLanded(st)
    # dt = pos_current
    # dt-1 = pos_old
    # vt = vel_current
    # vt-1 = vel_old
    # wt = angular_vel_current
    # wt-1 = angular_vel_old
    pos_distance = np.linalg.norm(pos_current - pos_old)
    vel_distance = np.linalg.norm(vel_current - vel_old)
    angular_vel_distance = angular_vel_current - angular_vel_old

    result = -100 * pos_distance - 100 * vel_distance - 100 * angular_vel_distance + has_landed

    print("Pos Distance ", pos_distance)
    print("Vel Distance ", vel_distance)
    print("Angular Vel Distance ", angular_vel_distance)
    print("has_landed", has_landed)
    print("Result ", result)


    return result


#enviroment für gym erstellen
#something like:
env = gym.make('LunarLander-v2')


observation = env.reset()


done = False

pos_old = np.array((0, 0))
vel_old = np.array((0, 0))
angular_vel_old = 0

pos_current = np.array((0, 0))
vel_current = np.array((0, 0))
angular_vel_current = 0

first_step = True

while not done:
    observation, reward, done, info = env.step(env.action_space.sample()) #action_space.sample lässt den moonlander einfach zufällig Aktionen ausführen
    # Position x and y
    pos_current = np.array((observation[0], observation[1]))

    # Velocity x and y
    vel_current = np.array((observation[2], observation[3]))

    # Angular Velocity x and y
    angular_vel_current = observation[5]

    contact_leg_one = observation[6]
    contact_leg_two = observation[7]

    has_landed = contact_leg_one + contact_leg_two

    if first_step:
        pos_old = pos_current
        vel_old = vel_current
        angular_vel_old = angular_vel_current
        first_step = False

    evaluate_fitness(pos_current, pos_old, vel_current, vel_old, angular_vel_current, angular_vel_old, has_landed)

    # Save the last state
    pos_old = pos_current
    vel_old = vel_current
    angular_vel_old = angular_vel_current

    env.render()
