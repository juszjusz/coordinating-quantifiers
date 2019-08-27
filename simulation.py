from __future__ import division  # force python 3 division in python 2

import argparse
import logging, sys
import pickle
import time
import dill

import matplotlib

matplotlib.use('Agg')
from agent import Population
from guessing_game import GuessingGame
from data import Data


# import cProfile


class Simulation:

    def __init__(self, params, step_offset, population):
        self.data = Data(params['population_size'])
        self.population = population
        self.step_offset = step_offset
        self.params = params

    def run(self):
        for step in range(self.params["steps"]):
            step_with_offset = step + self.step_offset
            logging.debug("\n------------\nSTEP %d" % step_with_offset)
            selected_pairs = self.population.select_pairs_per_round(self.params['population_size'] // 2)

            for speaker, hearer in selected_pairs:
                game = GuessingGame(self.params['is_stage7_on'])
                logging.debug("\nGAME(%d, %d)" % (speaker.id, hearer.id))
                game.play(speaker=speaker, hearer=hearer)
                logging.debug("Number of categories of Agent(%d): %d" % (speaker.id, len(speaker.get_categories())))
                logging.debug("Number of categories of Agent(%d): %d" % (hearer.id, len(hearer.get_categories())))

            with open("./simulation_results/data/step%d.p" % step_with_offset, "wb") as write_handle:
                dill.dump((parsed_params, step_with_offset, self.population), write_handle)

            self.data.store_ds(self.population.agents)
            self.data.store_cs(self.population.agents)
            self.data.store_matrices(self.population.agents)
            self.data.store_langs(self.population.agents)
            self.data.store_cats(self.population.agents)
            self.data.pickle(step, self.population.agents)
            self.data.plot_success(dt=self.params['discriminative_threshold'], step=step)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    parser = argparse.ArgumentParser(prog='quantifiers simulation')

    parser.add_argument('--population_size', '-p', help='population size', type=int, default=10)
    parser.add_argument('--discriminative_threshold', '-dt', help='discriminative threshold', type=float, default=.85)
    parser.add_argument('--delta_inc', '-dinc', help='delta increment', type=float, default=.1)
    parser.add_argument('--delta_dec', '-ddec', help='delta decrement', type=float, default=.1)
    parser.add_argument('--delta_inh', '-dinh', help='delta inhibition', type=float, default=.1)
    parser.add_argument('--alpha', '-a', help='forgetting rate', type=float, default=.1)
    parser.add_argument('--super_alpha', '-sa', help='complete forgetting of categories that have smaller weights',
                        type=float, default=.01)
    parser.add_argument('--beta', '-b', help='learning rate', type=float, default=1.)
    parser.add_argument('--steps', '-s', help='number of steps', type=int, default=15)
    parser.add_argument('--runs', '-r', help='number of runs', type=int, default=1)
    parser.add_argument('--is_stage7_on', '-s7', help='is stage seven of the game switched on', type=bool,
                        default=False)
    parser.add_argument('--load_simulation', '-l', help='load and rerun simulation from pickled simulation step', type=str)

    parsed_params = vars(parser.parse_args())

    if parsed_params['load_simulation']:
        pickled_simulation_file = parsed_params['load_simulation']
        logging.debug("loading pickled simulation from %s file", pickled_simulation_file)
        with open(pickled_simulation_file, 'rb') as read_handle:
            _, step, population = pickle.load(read_handle)
        simulation = Simulation(params=parsed_params, step_offset=step, population=population)
    else:
        population = Population(parsed_params)
        simulation = Simulation(params=parsed_params, step_offset=0, population=population)

    start_time = time.time()
    simulation.run()
    exec_time = time.time() - start_time

    logging.debug("simulation took %dsec (with params %s)" % (exec_time, parsed_params))
