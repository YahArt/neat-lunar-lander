
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
import visualize




runs_per_net = 2 #depends how env. starts, like if its a realy random initialisation, then you might want to give it chance to run 
#simulation_seconds = 60.0 <-- not needed as env. kills itself?


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config): #wichtiger teil, den wir anpassen müssen
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        env = gym.make('BipedalWalker-v3')
        #env = gym.make('CartPole-v1')
        
        observation = env.reset()
        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        done = False

        while not done:
            #action = np.argmax(net.activate(observation)) #take action based on observation
            action = net.activate(observation)
            observation, reward, done, info = env.step(action) #action von oben ausführen, also das aus dem net? net.activate entspricht in etwas dem predict aus anderen deep/ml algo
            fitness += reward

        fitnesses.append(fitness)

    return np.mean(fitnesses)


def eval_genomes(genomes, config): #ändert sich eigentlich nie
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run(): #ändert sich eigentlich nie
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
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
    with open('winner-feedforward', 'wb') as f:
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
