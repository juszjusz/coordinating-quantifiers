from __future__ import division  # force python 3 division in python 2

import argparse
import logging, sys
import time

import matplotlib

matplotlib.use('Agg')
from agent import Population
from guessing_game import GuessingGame
from data import Data


# import cProfile


class Simulation:

    def __init__(self, params):
        self.params = params
        self.data = Data(params['population_size'])

    def run(self):

        population = Population(self.params)

        for step in range(self.params["steps"]):
            logging.debug("\n------------\nSTEP %d" % step)
            selected_pairs = population.select_pairs_per_round(self.params['population_size'] // 2)

            for speaker, hearer in selected_pairs:
                game = GuessingGame(self.params['is_stage7_on'])
                logging.debug("\nGAME(%d, %d)" % (speaker.id, hearer.id))
                game.play(speaker=speaker, hearer=hearer)
                logging.debug("Number of categories of Agent(%d): %d" % (speaker.id, len(speaker.get_categories())))
                logging.debug("Number of categories of Agent(%d): %d" % (hearer.id, len(hearer.get_categories())))

            self.data.store_ds(population.agents)
            self.data.store_cs(population.agents)
            self.data.store_matrices(population.agents)
            self.data.store_langs(population.agents)
            self.data.store_cats(population.agents)
            self.data.pickle(step, population.agents)
            self.data.plot_success(step)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    parser = argparse.ArgumentParser(prog='quantifiers simulation')

    parser.add_argument('--population_size', '-p', help='population size', type=int, default=10)
    parser.add_argument('--discriminative_threshold', '-d', help='discriminative threshold', type=float, default=.90)
    parser.add_argument('--delta_inc', '-di', help='delta increment', type=float, default=.1)
    parser.add_argument('--delta_dec', '-dd', help='delta decrement', type=float, default=.1)
    parser.add_argument('--alpha', '-a', help='forgetting rate', type=float, default=.1)
    parser.add_argument('--super_alpha', '-sa', help='complete forgetting of categories that have smaller weights', type=float, default=.01)
    parser.add_argument('--beta', '-b', help='learning rate', type=float, default=1.)
    parser.add_argument('--steps', '-s', help='number of steps', type=int, default=15)
    parser.add_argument('--runs', '-r', help='number of runs', type=int, default=1)
    parser.add_argument('--is_stage7_on', '-s7', help='is stage seven of the game switched on', type=bool, default=True)

    parsed_params = vars(parser.parse_args())

    start_time = time.time()
    Simulation(params=parsed_params).run()
    exec_time = time.time() - start_time
    logging.debug("simulation took %dsec (with params %s)" % (exec_time, parsed_params))

