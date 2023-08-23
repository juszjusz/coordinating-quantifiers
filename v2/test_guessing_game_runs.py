import json
import unittest

from v2.new_guessing_game import NewAgent, run_simulation, GameParams, new_random_f, RandomFunction


class TestGuessingGameWithNumericStimulus(unittest.TestCase):
    def setUp(self) -> None:
        with open('serialized_state_numeric.json', 'r', encoding='utf-8') as fj:
            game_state = json.load(fj)
            self.params = game_state['params']
            self.expected_population = game_state['population']

    def test_result(self):
        game_params = GameParams(**self.params)

        rf: RandomFunction = next(new_random_f(seed=game_params.seed))
        actual_population = run_simulation(game_params,
                                           rf.shuffle_list_random_function(),
                                           rf.flip_a_coin_random_function(),
                                           rf.pick_element_random_function()
                                           )

        actual_population = [NewAgent.to_dict(a) for a in actual_population]
        self.assertEqual(actual_population, self.expected_population)


class TestGuessingGameWithQuotientStimulus(unittest.TestCase):
    def setUp(self) -> None:
        with open('serialized_state_quotient.json', 'r', encoding='utf-8') as fj:
            game_state = json.load(fj)
            self.params = game_state['params']
            self.expected_population = game_state['population']

    def test_result(self):
        game_params = GameParams(**self.params)

        rf: RandomFunction = next(new_random_f(seed=game_params.seed))
        actual_population = run_simulation(game_params,
                                           rf.shuffle_list_random_function(),
                                           rf.flip_a_coin_random_function(),
                                           rf.pick_element_random_function()
                                           )

        actual_population = [NewAgent.to_dict(a) for a in actual_population]
        self.assertEqual(actual_population, self.expected_population)
