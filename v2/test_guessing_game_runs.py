import json
import unittest

from new_guessing_game import run_simulation, new_random_f, RandomFunctions
from domain_objects import GameParams, NewAgent
from calculator import NumericCalculator, QuotientCalculator


class TestGuessingGameWithNumericStimulus(unittest.TestCase):
    def setUp(self) -> None:
        with open('serialized_state_numeric.json', 'r', encoding='utf-8') as fj:
            game_state = json.load(fj)
            self.params = game_state['params']
            self.expected_population = game_state['population']

    def test_result(self):
        game_params = GameParams(**self.params)
        calculator = {'numeric': NumericCalculator.load_from_file(),
                      'quotient': QuotientCalculator.load_from_file()}[game_params.stimulus]

        rf: RandomFunctions = next(new_random_f(seed=game_params.seed))
        actual_population, _, _, _ = run_simulation(calculator, game_params,
                                           rf.shuffle_list_random_function(),
                                           rf.flip_a_coin_random_function(),
                                           rf.pick_element_random_function()
                                           )

        actual_population = [NewAgent.to_dict(a) for a in actual_population]

        actual_agent = actual_population[0]
        expected_agent = self.expected_population[0]
        actual_lxc = actual_agent['lxc']
        expected_lxc = expected_agent['lxc']
        self.assertEqual(actual_agent, expected_agent)
        self.assertEqual(actual_population, self.expected_population)


class TestGuessingGameWithQuotientStimulus(unittest.TestCase):
    def setUp(self) -> None:
        with open('serialized_state_quotient.json', 'r', encoding='utf-8') as fj:
            game_state = json.load(fj)
            self.params = game_state['params']
            self.expected_population = game_state['population']


    def test_result(self):
        game_params = GameParams(**self.params)
        calculator = {'numeric': NumericCalculator.load_from_file(),
                      'quotient': QuotientCalculator.load_from_file()}[game_params.stimulus]

        rf: RandomFunctions = next(new_random_f(seed=game_params.seed))
        actual_population, _, _, _ = run_simulation(calculator, game_params,
                                           rf.shuffle_list_random_function(),
                                           rf.flip_a_coin_random_function(),
                                           rf.pick_element_random_function()
                                           )

        actual_population = [NewAgent.to_dict(a) for a in actual_population]
        self.assertEqual(actual_population, self.expected_population)
