import gym
import numpy as np
import time

def evaluate_fitness(pos_current, pos_old, vel_current, vel_old, angular_vel_current, angular_vel_old, has_landed, is_using_main_engine):
    # −100 ∗ (dt − dt−1) − 100 ∗ (vt − vt−1) −100 ∗ (ωt − ωt−1) + hasLanded(st)
    # dt = pos_current
    # dt-1 = pos_old
    # vt = vel_current
    # vt-1 = vel_old
    # wt = angular_vel_current
    # wt-1 = angular_vel_old
    landed_reward = 0
    using_main_engine_punishment = 0
    pos_distance = np.linalg.norm(pos_current - pos_old)
    vel_distance = np.linalg.norm(vel_current - vel_old)
    angular_vel_distance = angular_vel_current - angular_vel_old

    if has_landed:
        landed_reward = 100

    if is_using_main_engine:
        using_main_engine_punishment = -2

    result = -100 * pos_distance - 100 * vel_distance -100 * angular_vel_distance + landed_reward + using_main_engine_punishment

    print("Pos Current X/Y ", pos_current)
    print("Pos Old X/Y ", pos_old)


    print("Vel Current X/Y ", vel_current)
    print("Vel Old X/Y ", vel_old)


    print("Pos Distance ", pos_distance)
    print("Vel Distance ", vel_distance)
    print("Angular Vel Distance ", angular_vel_distance)
    print("Landed Reward ", landed_reward)

    print("Using Main Engine: ", is_using_main_engine)
    print("Result ", result)

    return result


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
    time.sleep(0.015)
    current_action = env.action_space.sample()
    observation, reward, done, info = env.step(current_action)
    
    # Actions go from 0 to 3
    # 0: do nothing
    # 1: fire left orientation engine
    # 2: fire main engine
    # 3: fire right orientation engine
    is_using_main_engine = current_action == 2

    # Position x and y
    pos_current = np.array((observation[0], observation[1]))

    # Velocity x and y
    vel_current = np.array((observation[2], observation[3]))

    # Angular Velocity x and y
    angular_vel_current = observation[5]

    contact_leg_one = observation[6] == 1.0
    contact_leg_two = observation[7] == 1.0

    has_landed = contact_leg_one and contact_leg_two


    if first_step:
        pos_old = pos_current
        vel_old = vel_current
        angular_vel_old = angular_vel_current
        first_step = False

    evaluate_fitness(pos_current, pos_old, vel_current, vel_old, angular_vel_current, angular_vel_old, has_landed, is_using_main_engine)

    # Save the last state
    pos_old = pos_current
    vel_old = vel_current
    angular_vel_old = angular_vel_current

    env.render()
env.close()
