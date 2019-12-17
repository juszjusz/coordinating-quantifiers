from __future__ import division  # force python 3 division in python 2

import argparse
import logging, sys
import pickle
import time
from multiprocessing import Process

import dill

import matplotlib
from pathlib import Path

from path_provider import PathProvider
from inmemory_calculus import load_inmemory_calculus, inmem
import os
import shutil

from stimulus import QuotientBasedStimulusFactory, ContextFactory, NumericBasedStimulusFactory

matplotlib.use('Agg')
from agent import Population
from guessing_game import GuessingGame

class Simulation(Process):

    def __init__(self, params, step_offset, population, context_constructor, num, path_provider):
        super(Simulation, self).__init__()
        self.num = num
        self.path_provider = path_provider
        self.population = population
        self.step_offset = step_offset
        self.params = params
        self.context_constructor = context_constructor

    def run(self):

        start_time = time.time()
        for step in range(self.params["steps"]):
            step_with_offset = step + self.step_offset
            logging.debug("\n------------\nSTEP %d" % step_with_offset)
            selected_pairs = self.population.select_pairs_per_round(self.population.population_size // 2)

            for speaker, hearer in selected_pairs:
                game = GuessingGame(self.params['guessing_game_2'], self.context_constructor())
                logging.debug("\nGAME(%d, %d)" % (speaker.id, hearer.id))
                game.play(speaker=speaker, hearer=hearer)
                logging.debug("Number of categories of Agent(%d): %d" % (speaker.id, len(speaker.get_categories())))
                logging.debug("Number of categories of Agent(%d): %d" % (hearer.id, len(hearer.get_categories())))

            self.population.update_metrics()

            serialized_step_path = str(self.path_provider.get_simulation_step_path(step_with_offset))
            with open(serialized_step_path, "wb") as write_handle:
                dill.dump((step_with_offset, self.population), write_handle)

        exec_time = time.time() - start_time
        logging.debug("simulation {} took {}sec (with params {})".format(self.num, exec_time, self.params))


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    parser = argparse.ArgumentParser(prog='quantifiers simulation')

    parser.add_argument('--simulation_name', '-sn', help='simulation name', type=str, default='test')
    parser.add_argument('--population_size', '-p', help='population size', type=int, default=4)
    parser.add_argument('--stimulus', '-stm', help='quotient or numeric', type=str, default='quotient')
    parser.add_argument('--max_num', '-mn', help='max number for numerics or max denominator for quotients', type=int, default=100)
    parser.add_argument('--discriminative_threshold', '-dt', help='discriminative threshold', type=float, default=.95)
    parser.add_argument('--delta_inc', '-dinc', help='delta increment', type=float, default=.1)
    parser.add_argument('--delta_dec', '-ddec', help='delta decrement', type=float, default=.1)
    parser.add_argument('--delta_inh', '-dinh', help='delta inhibition', type=float, default=.1)
    parser.add_argument('--alpha', '-a', help='forgetting rate', type=float, default=.1)
    parser.add_argument('--super_alpha', '-sa', help='complete forgetting of categories that have smaller weights',
                        type=float, default=.01)
    parser.add_argument('--beta', '-b', help='learning rate', type=float, default=0.1)
    parser.add_argument('--steps', '-s', help='number of steps', type=int, default=200)
    parser.add_argument('--runs', '-r', help='number of runs', type=int, default=1)
    parser.add_argument('--guessing_game_2', '-gg2', help='is the second stage of the guessing game on', type=bool,
                        default=False)
    parser.add_argument('--load_simulation', '-l', help='load and rerun simulation from pickled simulation step',
                        type=str)
    parser.add_argument('--parallel', '-pl', help='run parallel runs', type=bool, default=True)

    parsed_params = vars(parser.parse_args())
    load_inmemory_calculus(parsed_params['stimulus'])

    stimulus_factory = None
    if parsed_params['stimulus'] == 'quotient':
        stimulus_factory = QuotientBasedStimulusFactory(inmem['STIMULUS_LIST'], parsed_params['max_num'])
    if parsed_params['stimulus'] == 'numeric':
        stimulus_factory = NumericBasedStimulusFactory(inmem['STIMULUS_LIST'], parsed_params['max_num'])
    context_constructor = ContextFactory(stimulus_factory)

    simulation_tasks = []
    if parsed_params['load_simulation']:
        for run in Path(parsed_params['load_simulation']).glob('*'):
            pickled_simulation_file = parsed_params['load_simulation']
            logging.debug("loading pickled simulation from {} file".format(pickled_simulation_file))
            with open(pickled_simulation_file, 'rb') as read_handle:
                step, population = pickle.load(read_handle)

        simulation_tasks.append(Simulation(params=parsed_params,
                                           step_offset=step + 1,
                                           population=population,
                                           context_constructor=context_constructor,
                                           num=0,
                                           path_provider=PathProvider.new_path_provider(parsed_params['simulation_name'])))
    else:
        simulation_path = os.path.abspath(parsed_params['simulation_name'])
        if os.path.exists(simulation_path):
            shutil.rmtree(simulation_path, ignore_errors=True)
        os.makedirs(simulation_path)
        os.makedirs(simulation_path + '/stats')

        for run in range(parsed_params['runs']):
            population = Population(parsed_params)
            root_path = Path(simulation_path).joinpath('run' + str(run))
            path_provider = PathProvider.new_path_provider(root_path)
            path_provider.create_directory_structure()
            simulation = Simulation(params=parsed_params,
                                    step_offset=0,
                                    population=population,
                                    context_constructor=context_constructor,
                                    num=run,
                                    path_provider=path_provider)
            # if parsed_params['parallel']:
            #     simulation_tasks.append(simulation)
            # else:
            simulation.run()

    path_provider = PathProvider.new_path_provider(parsed_params['simulation_name'])
    params_path = str(Path(path_provider.root_path).joinpath('params.p'))
    with open(params_path, 'wb') as write_params:
        dill.dump(parsed_params, write_params)

    in_mem_path = str(Path(path_provider.root_path).joinpath('inmem_calc.p'))
    with open(in_mem_path, 'wb') as write_inmem:
        dill.dump(inmem, write_inmem)

    stimuluses_path = str(Path(path_provider.root_path).joinpath('stimuluses.p'))
    with open(stimuluses_path, 'wb') as write_stimuluses:
        dill.dump(stimulus_factory.generate_all_stimuluses(), write_stimuluses)

    if parsed_params['parallel']:
        for simulation_task in simulation_tasks:
            simulation_task.start()
        for simulation_task in simulation_tasks:
            simulation_task.join()
