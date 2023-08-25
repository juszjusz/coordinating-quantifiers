import json
import argparse
import logging
from collections import Counter
from typing import List, Callable, Any, Tuple

from numpy.random import RandomState

import numpy as np
from tqdm import tqdm

from stats import confidence_intervals, means
from v2.calculator import NumericCalculator, QuotientCalculator
from v2.domain_objects import GameParams, NewAgent, AggregatedGameResultStats
from v2.game_graph import game_graph_with_stage_7, game_graph, GameGraph
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class RandomFunction:
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


def new_random_f(seed: int) -> RandomFunction:
    r = np.random.RandomState(seed=seed)
    while True:
        yield RandomFunction(r.randint(2 ** 31))


def select_speaker(speaker: NewAgent, _: NewAgent) -> NewAgent:
    return speaker


def select_hearer(_: NewAgent, hearer: NewAgent) -> NewAgent:
    return hearer


def run_simulation(game_params: GameParams, shuffle_list, flip_a_coin, pick_element) -> Tuple[List[NewAgent], Any, Any]:
    calculator = {'numeric': NumericCalculator.load_from_file(),
                  'quotient': QuotientCalculator.load_from_file()}[game_params.stimulus]

    def pair_partition(agents: List):
        return [agents[i:i + 2] for i in range(0, len(agents), 2)]

    context_constructor = calculator.context_factory(pick_element=pick_element)

    G: GameGraph = game_graph(flip_a_coin)
    assert game_params.population_size % 2 == 0, 'each agent must be paired'
    population = [NewAgent(agent_id, game_params) for agent_id in range(game_params.population_size)]
    stats = AggregatedGameResultStats(game_params)

    # buckets with counters
    bucket_size = 100
    states_sequences_cnts = {(i, i + bucket_size): Counter() for i in range(0, game_params.steps, bucket_size)}
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

            state_name = None
            action, agent_name, arg_names = G.start()
            args = [data_envelope[a] for a in arg_names]

            states_sequence = ['START']
            while state_name != 'NEXT_STEP':
                debug_msg = f'{state_name}, agent: {agent_name}, args: {args}'
                logger.debug(debug_msg)

                agent_selector = {'SPEAKER': select_speaker, 'HEARER': select_hearer}[agent_name]
                agent = agent_selector(speaker, hearer)

                state_name = action(stats, calculator, agent, context, data_envelope, *args)
                action, agent_name, arg_names = G(state_name)
                args = [data_envelope[a] for a in arg_names]
                states_sequence.append(state_name)
                logger.debug(data_envelope)

                states_cnts[step2bucket[step]].update([states_sequence[-1]])

            bucket = step2bucket[step]
            states_sequences_cnts[bucket].update(['->'.join(states_sequence)])

    return population, states_sequences_cnts, states_cnts


class PlotSuccessCommand:

    def __call__(self, runs, steps, agg_cs1, agg_cs2, agg_cs12, agg_ds):
        # shape steps x runs
        cs1_means = means(agg_cs1)
        cs2_means = means(agg_cs2)
        cs12_means = means(agg_cs12)
        ds_means = means(agg_ds)
        # nw_means = means(samples_nw)
        # logging.debug(nw_means)

        # nw_cis_l, nw_cis_u = confidence_intervals(samples_nw)
        # cs1_cis_l, cs1_cis_u = confidence_intervals(agg_cs1)
        # cs2_cis_l, cs2_cis_u = confidence_intervals(agg_cs2)
        # ds_cis_l, ds_cis_u = confidence_intervals(agg_ds)
        # cs12_cis_l, cs12_cis_u = confidence_intervals(agg_cs12)

        fig, ax1 = plt.subplots()
        plt.ylim(bottom=0)
        plt.ylim(top=100)
        plt.xlabel("step")
        plt.ylabel("success")

        plt.plot(range(steps), ds_means, 'r-', linewidth=0.6)
        for i in range(0, runs):
            plt.plot(range(steps), [agg_ds[s][i] for s in range(0, steps)], 'r-', linewidth=0.2, alpha=.3)

        plt.plot(range(steps), ds_means, 'g--', linewidth=0.6)
        for i in range(0, runs):
            plt.plot(range(steps), [ds_means[s][i] for s in range(0, steps)], 'g-', linewidth=0.2, alpha=.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel('|active lexicon|')

        # ax2.plot(x100, nw_means, 'b--', linewidth=0.3)
        # ax2.fill_between(x100, nw_cis_l, nw_cis_u,
        #                  color='b', alpha=.2)
        # ax2.set_yticks(range(0, 15, 1),
        #                ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'))
        # ax2.tick_params(axis='y')

        fig.tight_layout()
        # plt.savefig(str(succ_plot_path))
        plt.show()
        # plt.close()


def run_dummy_simulation(stimulus):
    game_params = {'population_size': 2, 'stimulus': stimulus, 'max_num': 100, 'discriminative_threshold': 0.95,
                   'discriminative_history_length': 50, 'delta_inc': 0.2, 'delta_dec': 0.2, 'delta_inh': 0.2,
                   'alpha': 0.01,
                   'super_alpha': 0.001, 'beta': 0.2, 'steps': 1000, 'runs': 1, 'guessing_game_2': False,
                   'seed': 0}
    p = GameParams(**game_params)

    rf: RandomFunction = next(new_random_f(p.seed))
    actual_population = run_simulation(p,
                                       rf.shuffle_list_random_function(),
                                       rf.flip_a_coin_random_function(),
                                       rf.pick_element_random_function()
                                       )
    game_state = {'params': game_params, 'population': [NewAgent.to_dict(agent) for agent in actual_population]}

    with open(f'serialized_state_{p.stimulus}.json', 'w', encoding='utf-8') as f:
        json.dump(game_state, f)


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
    parser.add_argument('--steps', '-s', help='number of steps', type=int, default=4000)
    parser.add_argument('--runs', '-r', help='number of runs', type=int, default=5)
    parser.add_argument('--guessing_game_2', '-gg2', help='is the second stage of the guessing game on',
                        action='store_true')
    parser.add_argument('--seed', help='set seed value to replicate a random values', type=int, default=1)

    parsed_params = vars(parser.parse_args())

    game_params = GameParams(**parsed_params)

    rf = new_random_f(seed=0)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # Set the handler level to DEBUG
    logger.addHandler(ch)


    def avg_series(elements: List, history=50):
        return [np.mean(elements[max(0, i - history):i]) for i in range(1, len(elements))]


    runs = game_params.runs
    agg_cs1 = []
    agg_ds = []
    for _, r in zip(range(runs), rf):
        population, states_sequences, states_cnts = run_simulation(game_params,
                                                                   r.shuffle_list_random_function(),
                                                                   r.flip_a_coin_random_function(),
                                                                   r.pick_element_random_function()
                                                                   )

        windowed_communicative_success1 = [avg_series(a.get_communicative_success1()) for a in population]
        windowed_communicative_success2 = [avg_series(a.get_communicative_success2()) for a in population]
        windowed_discriminative_success = [avg_series(a.get_discriminative_success()) for a in population]

        agg_cs1.append(np.mean(np.array(windowed_communicative_success1), axis=0))
        agg_ds.append(np.mean(np.array(windowed_discriminative_success), axis=0))
        # agg_communicative_success2 = [avg_series(a.get_communicative_success2()) for a in population]

        # print([len(NewAgent.to_dict(a)['categories']) for a in population])
        # print([len(NewAgent.to_dict(a)['words']) for a in population])
        # print(states_sequences)
        # print(states_cnts)
        # print(NewAgent.lxc_to_ndarray(population[1]))

    samples = np.array(agg_cs1).transpose() * 100
    PlotSuccessCommand()(runs, game_params.steps, samples, samples, samples, samples)
