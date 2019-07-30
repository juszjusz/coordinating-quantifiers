from __future__ import division  # force python 3 division in python 2

import argparse
import logging
import sys
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from agent import Population
from guessing_game import GuessingGame
from data import Data


# import cProfile

class Simulation:

    def __init__(self, params):
        self.params = params
        self.data = Data(self.params['population_size'])

    def run(self):

        population = Population(self.params['population_size'])

        for step in range(self.params["steps"]):
            logging.debug("\n------------\nSTEP %d" % step)
            selected_pairs = population.select_pairs_per_round(self.params['population_size'] // 2)

            for speaker, hearer in selected_pairs:
                game = GuessingGame(is_stage7_on=self.params['is_stage7_on'])
                logging.debug("\nGAME(%d, %d)" % (speaker.id, hearer.id))
                result = game.play(speaker=speaker, hearer=hearer)

                logging.debug("Number of categories of Agent(%d): %d" % (speaker.id, len(speaker.get_categories())))
                logging.debug("Number of categories of Agent(%d): %d" % (hearer.id, len(hearer.get_categories())))
                self.data.store_cs(result)

            self.data.store_ds(population.agents)
            self.data.store_matrices(population.agents)
            self.data.store_langs(population.agents)
            self.data.store_cats(population.agents)
            self.data.pickle(step, population.agents)
            self.data.plot_success(step)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    parser = argparse.ArgumentParser(prog='quantifiers simulation')

    parser.add_argument('--population_size', '-p', help='population size', type=int, default=10)
    parser.add_argument('--learning_rate', '-l', help='learning rate', type=float, default=0)
    parser.add_argument('--discriminative_threshold', '-d', help='discriminative threshold', type=float, default=.95)
    parser.add_argument('--weight_decay', '-w', help='weight decay', type=float, default=.1)
    parser.add_argument('--steps', '-s', help='number of steps', type=int, default=15)
    parser.add_argument('--runs', '-r', help='number of runs', type=int, default=1)
    parser.add_argument('--is_stage7_on', '-s7', help='is stage seven of the game switched on', type=bool, default=True)

    parsed_params = vars(parser.parse_args())

    start_time = time.time()
    Simulation(params=parsed_params).run()
    exec_time = time.time() - start_time
    logging.debug("simulation took %dsec (with params %s)" % (exec_time, parsed_params))
