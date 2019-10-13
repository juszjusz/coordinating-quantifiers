from __future__ import division  # force python 3 division in python 2

import argparse
import logging, sys
import pickle
import time
import dill

import matplotlib

from stimulus import context_factory

import os
matplotlib.use('Agg')
from agent import Population
from guessing_game import GuessingGame
from data import Data

# import cProfile


class Simulation:

    def __init__(self, params, step_offset, population, context_constructor, num):
        self.num = num
        self.data = Data(params['population_size'])
        self.population = population
        self.step_offset = step_offset
        self.params = params
        self.context_constructor = context_constructor

    def run(self):
        for step in range(self.params["steps"]):
            step_with_offset = step + self.step_offset
            logging.debug("\n------------\nSTEP %d" % step_with_offset)
            selected_pairs = self.population.select_pairs_per_round(self.population.population_size // 2)

            for speaker, hearer in selected_pairs:
                game = GuessingGame(self.params['is_stage7_on'], self.context_constructor())
                logging.debug("\nGAME(%d, %d)" % (speaker.id, hearer.id))
                game.play(speaker=speaker, hearer=hearer)
                logging.debug("Number of categories of Agent(%d): %d" % (speaker.id, len(speaker.get_categories())))
                logging.debug("Number of categories of Agent(%d): %d" % (hearer.id, len(hearer.get_categories())))

            with open("./%s/data_%d/step%d.p" % (self.params['simulation_name'], self.num, step_with_offset), "wb") as write_handle:
                dill.dump((parsed_params, step_with_offset, self.population), write_handle)

            self.population.update_cs()
            self.population.update_ds()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    parser = argparse.ArgumentParser(prog='quantifiers simulation')

    parser.add_argument('--simulation_name', '-sn', help='simulation name', type=str, default='simulation')
    parser.add_argument('--population_size', '-p', help='population size', type=int, default=10)
    parser.add_argument('--stimulus', '-stm', help='quotient or stimulus', type=str, default='quotient')
    parser.add_argument('--discriminative_threshold', '-dt', help='discriminative threshold', type=float, default=.85)
    parser.add_argument('--delta_inc', '-dinc', help='delta increment', type=float, default=.1)
    parser.add_argument('--delta_dec', '-ddec', help='delta decrement', type=float, default=.1)
    parser.add_argument('--delta_inh', '-dinh', help='delta inhibition', type=float, default=.1)
    parser.add_argument('--alpha', '-a', help='forgetting rate', type=float, default=.1)
    parser.add_argument('--super_alpha', '-sa', help='complete forgetting of categories that have smaller weights',
                        type=float, default=.01)
    parser.add_argument('--beta', '-b', help='learning rate', type=float, default=1.)
    parser.add_argument('--steps', '-s', help='number of steps', type=int, default=15)
    parser.add_argument('--runs', '-r', help='number of runs', type=int, default=2)
    parser.add_argument('--is_stage7_on', '-s7', help='is stage seven of the game switched on', type=bool,
                        default=False)
    parser.add_argument('--load_simulation', '-l', help='load and rerun simulation from pickled simulation step', type=str)

    parsed_params = vars(parser.parse_args())

    context_constructor = context_factory[parsed_params['stimulus']]

    if parsed_params['load_simulation']:
        pickled_simulation_file = parsed_params['load_simulation']
        logging.debug("loading pickled simulation from {} file".format(pickled_simulation_file))
        with open(pickled_simulation_file, 'rb') as read_handle:
            _, step, population = pickle.load(read_handle)
        simulation = Simulation(params=parsed_params, step_offset=step+1, population=population, context_constructor=context_constructor)
    else:

        cwd = os.getcwd()
        os.mkdir('%s/%s' % (cwd, parsed_params['simulation_name']))
        for r in range(parsed_params['runs']):
            os.mkdir('%s/%s/%s_%d' % (cwd, parsed_params['simulation_name'], 'data', r))

        for r in range(parsed_params['runs']):
            population = Population(parsed_params)
            simulation = Simulation(params=parsed_params,
                                    step_offset=0,
                                    population=population,
                                    context_constructor=context_constructor,
                                    num=r)
            start_time = time.time()
            simulation.run()
            exec_time = time.time() - start_time
            logging.debug("simulation {} took {}sec (with params {})".format(simulation.num, exec_time, parsed_params))
