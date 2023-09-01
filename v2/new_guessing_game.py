import json
import argparse
import logging
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import List, Callable, Any, Tuple

import networkx as nx
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from numpy.random import RandomState
import numpy as np
from tqdm import tqdm

from stats import confidence_intervals
from calculator import NumericCalculator, QuotientCalculator, Calculator
from domain_objects import GameParams, NewAgent, NewCategory
from game_graph import game_graph, GameGraph
import matplotlib.pyplot as plt

from plot_utils import plot_successes, plot_category

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class RandomFunctions:
    def __init__(self, seed: int):
        self._r = np.random.RandomState(seed=seed)

    def flip_a_coin_random_function(self) -> Callable[[], int]:
        def flip_a_coin() -> int:
            return self._r.binomial(1, .5)

        return flip_a_coin

    def shuffle_list_random_function(self) -> Callable[[List], None]:
        def shuffle_list(l: List) -> None:
            return self._r.shuffle(l)

        return shuffle_list

    def pick_element_random_function(self) -> Callable[[List], Any]:
        def pick_random_value(l: List) -> Any:
            i = self._r.randint(len(l))
            return l[i]

        return pick_random_value


def new_random_f(seed: int) -> RandomFunctions:
    r = np.random.RandomState(seed=seed)
    while True:
        yield RandomFunctions(r.randint(2 ** 31))


def select_speaker(speaker: NewAgent, _: NewAgent) -> NewAgent:
    return speaker


def select_hearer(_: NewAgent, hearer: NewAgent) -> NewAgent:
    return hearer


def avg_series(elements: List, history=50) -> List:
    return [np.mean(elements[max(0, i - history):i]) for i in range(1, len(elements))]


def run_simulation(calculator: Calculator, game_params: GameParams, shuffle_list, flip_a_coin, pick_element) -> Tuple[
    List[NewAgent], Any, Any]:
    def pair_partition(agents: List):
        return [agents[i:i + 2] for i in range(0, len(agents), 2)]

    context_constructor = calculator.context_factory(pick_element=pick_element)

    G: GameGraph = game_graph(flip_a_coin)
    assert game_params.population_size % 2 == 0, 'each agent must be paired'
    population = [NewAgent(agent_id, calculator, game_params) for agent_id in range(game_params.population_size)]

    # buckets with counters
    bucket_size = 100
    states_sequences_cnts = {(i, i + bucket_size): Counter() for i in range(0, game_params.steps, bucket_size)}
    state_edges_cnts = {(i, i + bucket_size): Counter() for i in range(0, game_params.steps, bucket_size)}
    states_cnts = {(i, i + bucket_size): Counter() for i in range(0, game_params.steps, bucket_size)}
    step2bucket = {i: (int(i / bucket_size) * bucket_size, (int(i / bucket_size) + 1) * bucket_size) for i in
                   range(game_params.steps)}

    for step in tqdm(range(game_params.steps)):
        shuffle_list(population)
        paired_agents = pair_partition(population)

        for speaker, hearer in paired_agents:
            debug_msg = f'step {step}'
            logger.debug(debug_msg)
            context = context_constructor()

            data_envelope = {'topic': 0}

            state_name, action, agent_name, arg_names = G.start()
            args = [data_envelope[a] for a in arg_names]
            states_sequence = [state_name]

            while state_name != 'NEXT_STEP':
                debug_msg = f'{state_name}, agent: {agent_name}, args: {args}'
                logger.debug(debug_msg)

                agent_selector = {'SPEAKER': select_speaker, 'HEARER': select_hearer}[agent_name]
                agent = agent_selector(speaker, hearer)

                state_name = action(agent, context, data_envelope, *args)
                action, agent_name, arg_names = G(state_name)
                args = [data_envelope[a] for a in arg_names]
                states_sequence.append(state_name)
                logger.debug(data_envelope)

                states_cnts[step2bucket[step]].update([states_sequence[-1]])
            bucket = step2bucket[step]
            state_edges_cnts[bucket].update(
                [(states_sequence[i], states_sequence[i + 1]) for i in range(1, len(states_sequence) - 1)])
            states_sequences_cnts[bucket].update(['->'.join(states_sequence)])

    return population, states_sequences_cnts, state_edges_cnts, states_cnts


def run_dummy_simulation(stimulus):
    game_params = {'population_size': 2, 'stimulus': stimulus, 'max_num': 100, 'discriminative_threshold': 0.95,
                   'discriminative_history_length': 50, 'delta_inc': 0.2, 'delta_dec': 0.2, 'delta_inh': 0.2,
                   'alpha': 0.01,
                   'super_alpha': 0.001, 'beta': 0.2, 'steps': 3000, 'runs': 1, 'guessing_game_2': False,
                   'seed': 0}
    p = GameParams(**game_params)

    rf: RandomFunctions = next(new_random_f(p.seed))
    actual_population, _, _, _ = run_simulation(p,
                                                rf.shuffle_list_random_function(),
                                                rf.flip_a_coin_random_function(),
                                                rf.pick_element_random_function()
                                                )
    game_state = {'params': game_params, 'population': [NewAgent.to_dict(agent) for agent in actual_population]}

    with open(f'serialized_state_{p.stimulus}.json', 'w', encoding='utf-8') as f:
        json.dump(game_state, f)


class PlotMonotonicityCommand:

    # def __init__(self, root_paths, stimuluses, params):
    #     self.root_path2 = None
    #     self.root_path1 = Path(root_paths[0])
    #     if len(root_paths) > 1:
    #         self.root_path2 = Path(root_paths[1])
    #
    #     self.stimuluses = stimuluses
    #     self.params = params
    #     self.steps = [max(step * 100 - 1, 0) for step in range(1 + int(self.params['steps'] / 100))]
    #     self.mon_plot_path = Path('.').joinpath('monotonicity.pdf')
    #     # self.array1 = zeros((self.params['runs'], self.steps))
    #     self.mon_samples1 = []
    #     # self.array2 = zeros((self.params['runs'], self.steps))
    #     self.mon_samples2 = []
    #     self.mon_means1 = []
    #     self.mon_cis1_l = []
    #     self.mon_cis1_u = []
    #     self.mon_means2 = []
    #     self.mon_cis2_l = []
    #     self.mon_cis2_u = []
    #
    # def get_data(self):
    #     # logging.debug("Root path %s" % self.root_path1)
    #     for step in self.steps:
    #         # logging.debug("Processing step %d" % step)
    #         sample = []
    #         for run_num, run_path in enumerate(self.root_path1.glob('run[0-9]*')):
    #             # logging.debug("Processing %s, %s" % (run_num, run_path))
    #             # logging.debug("Processing %s" % "step" + str(step) + ".p")
    #             step, population = pickle.load(run_path.joinpath("data/step" + str(step) + ".p").open('rb'))
    #             sample.append(population.get_mon(self.stimuluses))
    #             # logging.debug("mon val %f" % sample[-1])
    #         self.mon_samples1.append(sample)
    #     # for step in range(self.params['steps']):
    #     #    self.mon_samples1.append(list(self.array1[:, max(step*100-1, 0)]))
    #
    #     if self.root_path2 is not None:
    #         logging.debug("Root path %s" % self.root_path2)
    #         for step in self.steps:
    #             logging.debug("Processing step %d" % step)
    #             sample = []
    #             for run_num, run_path in enumerate(self.root_path2.glob('run[0-9]')):
    #                 logging.debug("Processing %s, %s" % (run_num, run_path))
    #                 # for step_path in PathProvider(run_path).get_data_paths():
    #                 # logging.debug("Processing %s" % step_path)
    #                 step, population = pickle.load(run_path.joinpath("data/step" + str(step) + ".p").open('rb'))
    #                 # self.array2[run_num, step] = population.get_mon()
    #                 # logging.debug("mon val %f" % self.array2[run_num, step])
    #                 sample.append(population.get_mon(self.stimuluses))
    #             self.mon_samples2.append(sample)
    #             # for step in range(self.params['steps']):
    #             # self.mon_samples2.append(list(self.array2[:, step]))
    #
    # def compute_stats(self):
    #     logging.debug('in compute_stats')
    #     self.mon_means1 = means(self.mon_samples1)
    #     # logging.debug(len(self.mon_means1))
    #
    #     self.mon_cis1_l, self.mon_cis1_u = confidence_intervals(self.mon_samples1)
    #
    #     if self.root_path2 is not None:
    #         self.mon_means2 = means(self.mon_samples2)
    #         self.mon_cis2_l, self.mon_cis2_u = confidence_intervals(self.mon_samples2)

    def __call__(self, steps, mon_means):
        # x = range(1, self.params['steps'] + 1)
        mon_means1 = mon_means
        mon_means2 = mon_means
        mon_cis1_l, mon_cis1_u = confidence_intervals(mon_means1)
        mon_cis2_l, mon_cis2_u = confidence_intervals(mon_means1)

        x = steps
        plt.ylim(bottom=0)
        plt.ylim(top=100)
        plt.xlabel("step")
        plt.ylabel("monotonicity")

        # for r in range(self.runs):
        #    plt.plot(x, [y * 100.0 for y in self.array[r]], '-')

        plt.plot(range(x), mon_means1, 'r--', linewidth=0.3)
        plt.fill_between(range(x), mon_cis1_l, mon_cis1_u, color='r', alpha=.2)
        plt.plot(range(x), mon_means2, 'b--', linewidth=0.3)
        plt.fill_between(range(x), mon_cis2_l, mon_cis2_u, color='b', alpha=.2)
        plt.legend(['mon. no ANS', 'mon. ANS'], loc='best')

        plt.savefig('mon.png')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='quantifiers simulation')
    # parser.add_argument('--simulation_name', '-sn', help='simulation name', type=str, default='test')
    parser.add_argument('--population_size', '-p', help='population size', type=int, default=6)
    parser.add_argument('--stimulus', '-stm', help='quotient or numeric', type=str, default='quotient',
                        choices=['quotient', 'numeric'])
    parser.add_argument('--max_num', '-mn', help='max number for numerics or max denominator for quotients',
                        type=int,
                        default=100)
    parser.add_argument('--discriminative_threshold', '-dt', help='discriminative threshold', type=float,
                        default=.95)
    parser.add_argument('--discriminative_history_length',
                        help='max length of discriminative successes sequence per agent',
                        type=int, default=50)
    parser.add_argument('--delta_inc', '-dinc', help='delta increment', type=float, default=.2)
    parser.add_argument('--delta_dec', '-ddec', help='delta decrement', type=float, default=.2)
    parser.add_argument('--delta_inh', '-dinh', help='delta inhibition', type=float, default=.2)
    parser.add_argument('--alpha', '-a', help='forgetting rate', type=float, default=.01)
    parser.add_argument('--super_alpha', '-sa', help='complete forgetting of categories that have smaller weights',
                        type=float, default=.001)
    parser.add_argument('--beta', '-b', help='learning rate', type=float, default=0.2)
    parser.add_argument('--steps', '-s', help='number of steps', type=int, default=3000)
    parser.add_argument('--runs', '-r', help='number of runs', type=int, default=5)
    parser.add_argument('--guessing_game_2', '-gg2', help='is the second stage of the guessing game on',
                        action='store_true')
    parser.add_argument('--seed', help='set seed value to replicate a random values', type=int, default=1)

    parsed_params = vars(parser.parse_args())

    game_params = GameParams(**parsed_params)

    r = next(new_random_f(seed=10))

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # Set the handler level to DEBUG
    logger.addHandler(ch)

    runs = game_params.runs
    agg_cs1 = []
    agg_ds = []
    # for _, r in zip(range(runs), rf):
    calculator = {'numeric': NumericCalculator.load_from_file(),
                  'quotient': QuotientCalculator.load_from_file()}[game_params.stimulus]

    population, states_sequences, states_edges_cnts, states_cnts = run_simulation(calculator,
                                                                                  game_params,
                                                                                  r.shuffle_list_random_function(),
                                                                                  r.flip_a_coin_random_function(),
                                                                                  r.pick_element_random_function()
                                                                                  )
    states_edges_cnts_normalized = []
    for bucket, v in states_edges_cnts.items():
        # total_cnt_in_bucket = sum([cnt for _, cnt in v.items()])
        bucket_start, bucket_end = bucket
        total_cnt_in_bucket = (bucket_end - bucket_start) * 2
        normalized_cnts = {edge: round(cnt / total_cnt_in_bucket, 3) for edge, cnt in v.items()}
        states_edges_cnts_normalized.append((bucket, normalized_cnts))

    windowed_communicative_success1 = [avg_series(a.get_communicative_success1()) for a in population]
    windowed_communicative_success2 = [avg_series(a.get_communicative_success2()) for a in population]
    windowed_discriminative_success = [avg_series(a.get_discriminative_success()) for a in population]
    active_lexicon_size = [len(a.get_words()) for a in population]
    agent = population[0]
    recreated_agent_snapshots = NewAgent.recreate_from_history(agent_id=agent.agent_id, calculator=calculator,
                                                               game_params=game_params,
                                                               updates_history=agent.updates_history)
    recreated_agent = recreated_agent_snapshots[-1]
    print(NewAgent.to_dict(recreated_agent))
    # print(recreated_agent.get_discriminative_success() == agent.get_discriminative_success())
    # r_m = NewAgent.to_dict(recreated_agent)['lxc']
    # m = NewAgent.to_dict(agent)['lxc']
    # r_cats = NewAgent.to_dict(recreated_agent)['categories']
    # cats = NewAgent.to_dict(agent)['categories']
    # r_words = NewAgent.to_dict(recreated_agent)['words']
    # words = NewAgent.to_dict(agent)['words']
    # print(r_m == m)
    # print(r_cats == cats)
    # print(r_words == words)
    # meanings = agent.get_word_meanings(calculator=calculator)
    # for w, stimuli in meanings.items():
    #     print(w, w.originated_from_category)
    #     print([round(num / denum, 3) for num, denum in w.originated_from_category.reactive_units()])
    #     print([round(num / denum, 3) for num, denum in stimuli])

    # plt.show()

    # monotonicity = [a.get_monotonicity(calculator.stimuli(), calculator) for a in population]
    # print(monotonicity)
    # avg_monotonicity = np.mean(np.array(monotonicity), axis=0) * 100

    cs1 = np.mean(np.array(windowed_communicative_success1), axis=0) * 100
    ds = np.mean(np.array(windowed_discriminative_success), axis=0) * 100
    # agg_cs1.append(np.mean(np.array(windowed_communicative_success1), axis=0))
    # agg_cs1.append(np.mean(np.array(windowed_discriminative_success), axis=0))
    # agg_ds.append(np.mean(np.array(windowed_discriminative_success), axis=0))
    # agg_communicative_success2 = [avg_series(a.get_communicative_success2()) for a in population]

    # print([len(NewAgent.to_dict(a)['categories']) for a in population])
    # print([len(NewAgent.to_dict(a)['words']) for a in population])
    # print(states_sequences)
    # print(states_cnts)
    # print(NewAgent.lxc_to_ndarray(population[1]))

    # samples = np.array(agg_cs1).transpose() * 100
    # PlotSuccessCommand()(runs, game_params.steps, cs1, cs1, cs1, cs1)

    # plot_category('agent', calculator)
    # plot_successes(game_params.steps, list(cs1), list(cs1), list(cs1), list(ds))

    # meanings = population[0].compute_word_meanings(calculator)
    # print(meanings)
    # PlotMonotonicityCommand()(game_params.steps, monotonicity)
