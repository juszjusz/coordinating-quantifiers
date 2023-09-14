import json
import time
import unittest

from new_guessing_game import run_simulation, random_functions
from domain_objects import GameParams, NewAgent
from calculator import load_stimuli_and_calculator


class TestGuessingGameWithNumericStimulus(unittest.TestCase):
    def setUp(self) -> None:
        with open('serialized_state_numeric.json', 'r', encoding='utf-8') as fj:
            game_state = json.load(fj)
            self.params = game_state['params']
            self.expected_population = game_state['population']

    def test_result(self):
        game_params = GameParams(**self.params)
        stimuli, calculator = load_stimuli_and_calculator(game_params.stimulus)

        actual_population = run_simulation(game_params.seed, stimuli, calculator, game_params)

        actual_population = [NewAgent.to_dict(a) for a in actual_population]

        actual_agent = actual_population[0]
        expected_agent = self.expected_population[0]
        actual_lxc = actual_agent['lxc']
        expected_lxc = expected_agent['lxc']
        self.assertEqual(actual_agent, expected_agent)
        # self.assertEqual(actual_population, self.expected_population)


class TestGuessingGameWithQuotientStimulus(unittest.TestCase):
    def setUp(self) -> None:
        with open('serialized_state_quotient.json', 'r', encoding='utf-8') as fj:
            game_state = json.load(fj)
            self.params = game_state['params']
            self.expected_population = game_state['population']

    def test_result(self):
        game_params = GameParams(**self.params)
        stimuli, calculator = load_stimuli_and_calculator(game_params.stimulus)

        actual_population = run_simulation(game_params.seed, stimuli, calculator, game_params)

        actual_population = [NewAgent.to_dict(a) for a in actual_population]
        self.assertEqual(actual_population, self.expected_population)


class TestSnapshot(unittest.TestCase):
    def test_snapshot_for_numeric(self):
        params = {'population_size': 2, 'stimulus': 'numeric', 'max_num': 100, 'discriminative_threshold': 0.95,
                  'discriminative_history_length': 50, 'delta_inc': 0.2, 'delta_dec': 0.2, 'delta_inh': 0.2,
                  'alpha': 0.01,
                  'super_alpha': 0.001, 'beta': 0.2, 'steps': 3000, 'runs': 1, 'guessing_game_2': False,
                  'seed': 0}

        game_params = GameParams(**params)
        stimuli, calculator = load_stimuli_and_calculator(game_params.stimulus)

        actual_population = run_simulation(game_params.seed, stimuli, calculator, game_params)

        agent0: NewAgent = actual_population[0]
        recreated_agent0_snapshots = NewAgent.recreate_from_history(agent_id=agent0.agent_id, calculator=calculator,
                                                                    game_params=game_params,
                                                                    updates_history=agent0.updates_history)
        step, recreated_agent0 = recreated_agent0_snapshots[-1]
        source_agent_lxc = NewAgent.to_dict(agent0)['lxc']
        source_agent_ds = NewAgent.to_dict(agent0)['discriminative_success']
        recreated_agent_lxc = NewAgent.to_dict(recreated_agent0)['lxc']
        recreated_agent_ds = NewAgent.to_dict(recreated_agent0)['discriminative_success']

        self.assertEqual(source_agent_lxc, recreated_agent_lxc)
        # self.assertEqual(source_agent_ds, recreated_agent_ds)

    def test_snapshot_for_quotient(self):
        params = {'population_size': 2, 'stimulus': 'numeric', 'max_num': 100, 'discriminative_threshold': 0.95,
                  'discriminative_history_length': 50, 'delta_inc': 0.2, 'delta_dec': 0.2, 'delta_inh': 0.2,
                  'alpha': 0.01,
                  'super_alpha': 0.001, 'beta': 0.2, 'steps': 3000, 'runs': 1, 'guessing_game_2': False,
                  'seed': 0}

        game_params = GameParams(**params)
        stimuli, calculator = load_stimuli_and_calculator(game_params.stimulus)

        actual_population = run_simulation(game_params.seed, stimuli, calculator, game_params)

        agent0: NewAgent = actual_population[0]
        recreated_agent0_snapshots = NewAgent.recreate_from_history(agent_id=agent0.agent_id, calculator=calculator,
                                                                    game_params=game_params,
                                                                    updates_history=agent0.updates_history)
        step, recreated_agent0 = recreated_agent0_snapshots[-1]
        source_agent_lxc = NewAgent.to_dict(agent0)['lxc']
        source_agent_ds = NewAgent.to_dict(agent0)['discriminative_success']
        recreated_agent_lxc = NewAgent.to_dict(recreated_agent0)['lxc']
        recreated_agent_ds = NewAgent.to_dict(recreated_agent0)['discriminative_success']

        self.assertEqual(source_agent_lxc, recreated_agent_lxc)
        # self.assertEqual(source_agent_ds, recreated_agent_ds)


class TestTime(unittest.TestCase):

    def test_result(self):
        params = {'population_size': 4, 'stimulus': 'quotient', 'max_num': 100, 'discriminative_threshold': 0.95,
                  'discriminative_history_length': 50, 'delta_inc': 0.2, 'delta_dec': 0.2, 'delta_inh': 0.2,
                  'alpha': 0.01,
                  'super_alpha': 0.001, 'beta': 0.2, 'steps': 3000, 'runs': 1, 'guessing_game_2': False,
                  'seed': 0}

        game_params = GameParams(**params)
        stimuli, calculator = load_stimuli_and_calculator(game_params.stimulus)


        start = time.time()
        actual_population = run_simulation(game_params.seed, stimuli, calculator, game_params)
        elapsed = time.time() - start

        print(elapsed)