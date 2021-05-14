
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
import helpers.visualize
import math




runs_per_net = 2 #depends how env. starts, like if its a realy random initialisation, then you might want to give it chance to run 
#simulation_seconds = 60.0 <-- not needed as env. kills itself?

def evaluate_fitness(x_pos, contact_leg_one, contact_leg_two):
    fitness = 0.0
    
    # The middle is at 0.0
    penalty = abs(x_pos) * -1

    # Reward if lunar lander is pretty close to the middle
    bonus = x_pos > -0.2 and x_pos < 0.2

    fitness += penalty
    if bonus:
        fitness += 2
    elif bonus and contact_leg_one and contact_leg_two:
        fitness += 5

    return fitness


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

        while not done:
            action = np.argmax(net.activate(observation)) #take action based on observation
            observation, reward, done, info = env.step(action) #action von oben ausführen, also das aus dem net? net.activate entspricht in etwas dem predict aus anderen deep/ml algo
            x_pos = observation[0]
            contact_leg_one = observation[6] == 1.0
            contact_leg_two = observation[7] == 1.0
            fitness += evaluate_fitness(x_pos, contact_leg_one, contact_leg_two) 
        fitnesses.append(fitness)

    return np.mean(fitnesses)


def eval_genomes(genomes, config): #ändert sich eigentlich nie
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(): #ändert sich eigentlich nie
    # Load the config file, which is assumed to live in
    # the same directory as this script.
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

    print(winner)
"""
    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)

"""
if __name__ == '__main__':
    run()