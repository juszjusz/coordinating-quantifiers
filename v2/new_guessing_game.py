import json
import argparse
import logging
import math
from collections import Counter
from itertools import groupby
from multiprocessing import Pool
from typing import List, Callable, Any, Tuple

from numpy.random import RandomState
import numpy as np
from tqdm import tqdm
from calculator import Calculator, context_factory, Stimulus, load_stimuli_and_calculator
from domain_objects import GameParams, NewAgent
from game_graph import game_graph, GameGraph

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)  # Set the handler level to DEBUG
logger.addHandler(ch)


def flip_a_coin_random_function(seed: int) -> Callable[[], int]:
    random_state = np.random.RandomState(seed)

    def flip_a_coin() -> int:
        return random_state.binomial(1, .5)

    return flip_a_coin


def shuffle_list_random_function(seed: int) -> Callable[[List], None]:
    random_state = np.random.RandomState(seed)

    def shuffle_list(l: List) -> None:
        random_state.shuffle(l)

    return shuffle_list


def pick_element_random_function(seed: int) -> Callable[[List], Any]:
    random_state = np.random.RandomState(seed)

    def pick_random_value(l: List) -> Any:
        i = random_state.randint(len(l))
        return l[i]

    return pick_random_value


def random_functions(seed: int):
    """ Generate 3 functions with responsible for randomization in simulation. """
    r = np.random.RandomState(seed=seed)
    while True:
        seed0, seed1, seed2 = r.randint(2 ** 31, size=3)
        shuffle_list = shuffle_list_random_function(seed0)
        flip_a_coin = flip_a_coin_random_function(seed1)
        pick_element = pick_element_random_function(seed2)
        yield shuffle_list, flip_a_coin, pick_element


def select_speaker(speaker: NewAgent, _: NewAgent) -> NewAgent:
    return speaker


def select_hearer(_: NewAgent, hearer: NewAgent) -> NewAgent:
    return hearer


def avg_series(elements: List, history=50) -> List:
    return [np.mean(elements[max(0, i - history):i]) for i in range(1, len(elements))]


def recreate_from_history(agents: List[Tuple[int, NewAgent]], stimuli: List[Stimulus], calculator: Calculator, game_params: GameParams,
                          snapshot_rate: int):
    def compute_monotonicity_in_snapshot(step: int, agent_snapshot: NewAgent):
        if step > 0:
            word2activations = agent_snapshot.compute_word_meanings()
            monotonic_words_count = sum(
                NewAgent.is_monotone_new(activations) for _, activations in word2activations.items())
            return monotonic_words_count / len(word2activations)
        else:
            return 0

    def compute_convexity_in_snapshot(step: int, agent_snapshot: NewAgent):
        if step > 0:
            word2activations = agent_snapshot.compute_word_pragmatic_meanings(stimuli)
            monotonic_words_count = sum(
                NewAgent.is_convex_new(activations) for _, activations in word2activations.items())
            return monotonic_words_count / len(word2activations)
        else:
            return 0

    def compute_active_lexicon_in_snapshot(step: int, agent_snapshot: NewAgent):
        if step > 0:
            return agent_snapshot.compute_active_words()
        else:
            return []

    snapshots = []
    for run, agent in agents:
        agent_snapshots = NewAgent.recreate_from_history(agent_id=agent.agent_id, calculator=calculator,
                                                         game_params=game_params,
                                                         updates_history=agent.updates_history,
                                                         snapshot_rate=snapshot_rate)

        active_lexicon_in_snapshots = [compute_active_lexicon_in_snapshot(step, agent_snapshot)
                                       for step, agent_snapshot in agent_snapshots]
        monotonicity_in_snapshots = [compute_monotonicity_in_snapshot(step, agent_snapshot) for step, agent_snapshot in
                                     agent_snapshots]
        convexity_in_snapshots = [compute_convexity_in_snapshot(step, agent_snapshot) for step, agent_snapshot in
                                  agent_snapshots]

        agent_snapshots = [(step, agent_snapshot, active_lexicon_snapshot, monotonicity_snapshot, convexity_snapshot)
                           for
                           ((step, agent_snapshot), active_lexicon_snapshot, monotonicity_snapshot, convexity_snapshot)
                           in zip(agent_snapshots, active_lexicon_in_snapshots, monotonicity_in_snapshots,
                                  convexity_in_snapshots)]

        snapshots.append((run, agent_snapshots))

    return snapshots


def recreate_agents_snapshots_in_parallel(populations: List[List[NewAgent]], stimuli: List[Stimulus], calculator: Calculator,
                                          game_params: GameParams, snapshot_rate=200, processes_num=12):
    flatten_populations = [(run, agent) for run, population in enumerate(populations) for agent in population]
    bucket_size = math.ceil(len(flatten_populations) / processes_num)
    bucketed_agents = [flatten_populations[i:i + bucket_size] for i in range(0, len(flatten_populations), bucket_size)]
    args = [(bucket, stimuli, calculator, game_params, snapshot_rate) for bucket in bucketed_agents]
    with Pool(processes=processes_num) as pool:
        agent_snapshots = pool.starmap(recreate_from_history, args)
    agent_snapshots = [snapshots for bucket in agent_snapshots for snapshots in bucket]
    agent_snapshots.sort(key=lambda run2agent: run2agent[0])
    snapshots_grouped_by_populations = [[*v] for k, v in groupby(agent_snapshots, key=lambda run2agent: run2agent[0])]
    snapshots_grouped_by_populations = [[snapshots for run, snapshots in population] for population in
                                        snapshots_grouped_by_populations]
    return snapshots_grouped_by_populations


def run_simulations_in_parallel(stimuli: List[Stimulus], calculator: Calculator, game_params: GameParams, processes_num=8):
    r = RandomState(game_params.seed)

    with Pool(processes=processes_num) as pool:
        seeds = [(run, r.randint(0, 2 ** 31)) for run in range(game_params.runs)]
        args = [(seed, stimuli, calculator, game_params, run + 1) for run, seed in seeds]
        populations = pool.starmap(run_simulation, args)

    return populations


def run_simulation(seed: int, stimuli: List[Stimulus], calculator: Calculator, game_params: GameParams, run=0):
    r_functions = random_functions(seed=seed)

    shuffle_list, flip_a_coin, pick_element = next(r_functions)

    def pair_partition(agents: List):
        return [agents[i:i + 2] for i in range(0, len(agents), 2)]

    context_constructor = context_factory(stimuli=stimuli, pick_element=pick_element)

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

    for step in tqdm(range(game_params.steps), position=run):
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

    return population


def run_dummy_simulation(stimulus):
    game_params = {'population_size': 2, 'stimulus': stimulus, 'max_num': 100, 'discriminative_threshold': 0.95,
                   'discriminative_history_length': 50, 'delta_inc': 0.2, 'delta_dec': 0.2, 'delta_inh': 0.2,
                   'alpha': 0.01,
                   'super_alpha': 0.001, 'beta': 0.2, 'steps': 3000, 'runs': 1, 'guessing_game_2': False,
                   'seed': 0}
    params = GameParams(**game_params)

    stimuli, calculator = load_stimuli_and_calculator(params.stimulus)

    shuffle_list, flip_a_coin, pick_element = next(random_functions(seed=0))

    actual_population = run_simulation(stimuli, calculator, params, shuffle_list, flip_a_coin, pick_element)
    game_state = {'params': game_params, 'population': [NewAgent.to_dict(agent) for agent in actual_population]}

    with open(f'serialized_state_{params.stimulus}.json', 'w', encoding='utf-8') as f:
        json.dump(game_state, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='quantifiers simulation')
    # parser.add_argument('--simulation_name', '-sn', help='simulation name', type=str, default='test')
    parser.add_argument('--population_size', '-p', help='population size', type=int, default=6)
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
    parser.add_argument('--steps', '-s', help='number of steps', type=int, default=1000)
    parser.add_argument('--runs', '-r', help='number of runs', type=int, default=4)
    parser.add_argument('--guessing_game_2', '-gg2', help='is the second stage of the guessing game on',
                        action='store_true')
    parser.add_argument('--seed', help='set seed value to replicate a random values', type=int, default=1)

    parsed_params = vars(parser.parse_args())

    game_params = GameParams(**parsed_params)

    shuffle_list, flip_a_coin, pick_element = next(random_functions(seed=0))

    stimuli, calculator = load_stimuli_and_calculator(game_params.stimulus)

    population = run_simulation(0, stimuli, calculator, game_params)
    # states_edges_cnts_normalized = []
    # for bucket, v in states_edges_cnts.items():
    #     bucket_start, bucket_end = bucket
    #     total_cnt_in_bucket = (bucket_end - bucket_start) * 2
    #     normalized_cnts = {edge: round(cnt / total_cnt_in_bucket, 3) for edge, cnt in v.items()}
    #     states_edges_cnts_normalized.append((bucket, normalized_cnts))
    populations_snapshots = recreate_agents_snapshots_in_parallel(populations=[population], stimuli=stimuli, calculator=calculator, game_params=game_params)
    # population_snapshots = [
    #     NewAgent.recreate_from_history(agent_id=a.agent_id, calculator=calculator, game_params=game_params,
    #                                    updates_history=a.updates_history) for a in population]
    agent = population[0]
    agent_active_words = agent.compute_word_meanings()
    # w2meanings = agent.compute_word_pragmatic_meanings(stimuli)
    for word, activations in agent_active_words.items():
        print(NewAgent.is_monotone_new(activations))
    # is_word_monotone = {}
    # for w, meaning in w2meanings.items():
    #     is_word_monotone[w] = NewAgent.is_monotone_new(meaning)

    # print(recreated_agent.get_discriminative_success() == agent.get_discriminative_success())
    # meanings = agent.get_word_meanings(calculator=calculator)
    # for w, stimuli in meanings.items():
    #     print(w, w.originated_from_category)
    #     print([round(num / denum, 3) for num, denum in w.originated_from_category.reactive_units()])
    #     print([round(num / denum, 3) for num, denum in stimuli])

    # plt.show()
