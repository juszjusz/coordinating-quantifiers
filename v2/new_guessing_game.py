import json
import argparse
import logging
from typing import List, Callable, Any

from numpy.random import RandomState

import numpy as np

from v2.calculator import NumericCalculator, QuotientCalculator
from v2.domain_objects import GameParams, NewAgent, AggregatedGameResultStats
from v2.game_graph import game_graph_with_stage_7, select_speaker, select_hearer

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


def run_simulation(game_params: GameParams, shuffle_list, flip_a_coin, pick_element) -> List[NewAgent]:
    calculator = {'numeric': NumericCalculator.load_from_file(),
                  'quotient': QuotientCalculator.load_from_file()}[game_params.stimulus]

    def pair_partition(agents: List):
        return [agents[i:i + 2] for i in range(0, len(agents), 2)]

    context_constructor = calculator.context_factory(pick_element=pick_element)

    game_graph = game_graph_with_stage_7(flip_a_coin)
    assert game_params.population_size % 2 == 0, 'each agent must be paired'
    population = [NewAgent(agent_id, game_params) for agent_id in range(game_params.population_size)]

    stats = AggregatedGameResultStats(game_params)
    for step in range(game_params.steps):
        shuffle_list(population)
        paired_agents = pair_partition(population)

        for speaker, hearer in paired_agents:
            logger.debug(f'step {step}')
            context = context_constructor()

            data_envelope = {'topic': 0}

            state_name = '2_SPEAKER_DISCRIMINATION_GAME'
            state = game_graph[state_name]
            action = state['action']
            agent_name = state['agent']
            arg_names = state['args']
            args = [data_envelope[a] for a in arg_names]

            while state_name != 'NEXT_STEP':
                logger.debug(f'{state_name}, agent: {agent_name}, args: {args}')

                agent_selector = {'SPEAKER': select_speaker, 'HEARER': select_hearer}[agent_name]
                agent = agent_selector(speaker, hearer)

                state_name = action(stats, calculator, agent, context, data_envelope, *args)
                state = game_graph[state_name]
                action = state['action']
                agent_name = state['agent']
                arg_names = state['args']
                args = [data_envelope[a] for a in arg_names]

                logger.debug(data_envelope)

    return population


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
    run_dummy_simulation(stimulus='numeric')
    parser = argparse.ArgumentParser(prog='quantifiers simulation')

    # parser.add_argument('--simulation_name', '-sn', help='simulation name', type=str, default='test')
    parser.add_argument('--population_size', '-p', help='population size', type=int, default=2)
    parser.add_argument('--stimulus', '-stm', help='quotient or numeric', type=str, default='numeric',
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
    parser.add_argument('--steps', '-s', help='number of steps', type=int, default=250)
    parser.add_argument('--runs', '-r', help='number of runs', type=int, default=1)
    parser.add_argument('--guessing_game_2', '-gg2', help='is the second stage of the guessing game on',
                        action='store_true')
    parser.add_argument('--seed', help='set seed value to replicate a random values', type=int, default=1)

    parsed_params = vars(parser.parse_args())

    game_params = GameParams(**parsed_params)


    rf = new_random_f(seed=0)


    def avg_series(l: List, history=50):
        return [np.mean(l[-i - history:]) for i in range(len(l))]


    for _, r in zip(range(1), rf):
        population = run_simulation(game_params,
                                    r.shuffle_list_random_function(),
                                    r.flip_a_coin_random_function(),
                                    r.pick_element_random_function()
                                    )

        print([NewAgent.to_dict(a) for a in population])
        a = population[0]

        agg_communicative_success1 = [avg_series(a.get_communicative_success1()) for a in population]
        agg_communicative_success2 = [avg_series(a.get_communicative_success2()) for a in population]
    # print([len(NewAgent.to_dict(a)['categories']) for a in population])
    # print([len(NewAgent.to_dict(a)['words']) for a in population])
    # categories cnt after 1000x
    # [16, 13, 13, 21, 12, 29, 17, 14, 23, 11, 12, 34, 19, 24, 12, 28, 9, 31, 24, 24]
    # words cnt after 1000x
    # [91, 92, 98, 94, 89, 102, 95, 94, 105, 95, 97, 98, 92, 93, 95, 101, 96, 102, 102, 88]

