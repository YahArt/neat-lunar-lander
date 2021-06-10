
"""
Quellen: 
    https://www.youtube.com/watch?v=ZC0gMhYhwW0&t=804s
    https://github.com/CodeReclaimers/neat-python/tree/master/examples/openai-lander

"""
import multiprocessing
import os
import pickle
import numpy as np
import neat
import gym 
import Box2D
import math




runs_per_net = 2 #depends how env. starts, like if its a realy random initialisation, then you might want to give it chance to run 
#simulation_seconds = 60.0 <-- not needed as env. kills itself?

def evaluate_fitness(pos_current, pos_old, vel_current, vel_old, angular_vel_current, angular_vel_old, has_landed, is_using_main_engine):
    # https://arxiv.org/pdf/2011.11850.pdf
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
        using_main_engine_punishment = 0.5

    return -100 * pos_distance - 100 * vel_distance -100 * angular_vel_distance + landed_reward -100 * using_main_engine_punishment

def calculate_vector_distance(a, b):
    return np.linalg.norm(a - b)

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config): #wichtiger teil, den wir anpassen müssen
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        env = gym.make('LunarLander-v2')
        
        observation = env.reset()
        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        done = False

        pos_old = np.array((0, 0))
        vel_old = np.array((0, 0))
        angular_vel_old = 0

        pos_current = np.array((0, 0))
        vel_current = np.array((0, 0))
        angular_vel_current = 0

        first_step = True

        while not done:
            action = np.argmax(net.activate(observation)) #take action based on observation
            observation, reward, done, info = env.step(action)
            
            # Actions go from 0 to 3
            # 0: do nothing
            # 1: fire left orientation engine
            # 2: fire main engine
            # 3: fire right orientation engine
            is_using_main_engine = action == 2

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

            fitness += evaluate_fitness(pos_current, pos_old, vel_current, vel_old, angular_vel_current, angular_vel_old, has_landed, is_using_main_engine)

            # Save the last state
            pos_old = pos_current
            vel_old = vel_current
            angular_vel_old = angular_vel_current
        fitnesses.append(fitness)

    return np.mean(fitnesses)


def eval_genomes(genomes, config): #ändert sich eigentlich nie
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(): #ändert sich eigentlich nie
    # Load the config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config/config-feedforward.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner/winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

if __name__ == '__main__':
    run()
