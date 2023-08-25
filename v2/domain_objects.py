from __future__ import division  # force python 3 division in python 2

import dataclasses
import logging
from collections import deque
from threading import Lock
from typing import Tuple,  Callable, List, Dict, Union

import numpy as np
from numpy import ndarray

from v2.calculator import Calculator, NewAbstractStimulus

logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)

class NewCategory:
    def __init__(self, category_id: int):
        self.category_id = category_id
        self.is_active = True
        self._weights = []
        self._reactive_units = []

    def __hash__(self):
        return self.category_id

    def __repr__(self):
        return f'[{self._weights}x{self._reactive_units}]'

    def __eq__(self, o) -> bool:
        if not isinstance(o, NewCategory):
            return False

        return self.category_id == o.category_id

    def reactive_units(self):
        return self._reactive_units

    def weights(self):
        return self._weights

    def response(self, stimulus: NewAbstractStimulus, calculator: Calculator):
        return sum([weight * calculator.dot_product(ru_value, stimulus.value()) for weight, ru_value in
                    zip(self._weights, self._reactive_units)])

    def add_reactive_unit(self, stimulus: NewAbstractStimulus, weight=0.5):
        self._weights.append(weight)
        self._reactive_units.append(stimulus.value())

    def select(self, context: Tuple[NewAbstractStimulus, NewAbstractStimulus], calculator: Calculator) -> int or None:
        s1, s2 = context
        r1, r2 = self.response(s1, calculator), self.response(s2, calculator)
        if r1 == r2:
            return None
        else:
            return np.argmax([r1, r2])

    def reinforce(self, stimulus: NewAbstractStimulus, beta, calculator: Calculator):
        self._weights = [weight + beta * calculator.dot_product(ru, stimulus.value()) for weight, ru in
                         zip(self._weights, self._reactive_units)]

    def decrement_weights(self, alpha):
        self._weights = [weight - alpha * weight for weight in self._weights]

    def max_weigth(self):
        return max(self._weights)

    def discretized_distribution(self, calculator: Calculator):
        return self.__apply_fun_to_coordinates(lambda x: np.sum(x, axis=0), calculator)

    def union(self, calculator: Calculator):
        return self.__apply_fun_to_coordinates(lambda x: np.max(x, axis=0), calculator)

    # Given values f(x0),f(x1),...,f(xn); g(x0),g(x1),...,g(xn) for functions f, g defined on points x0 < x1 < ... < xn
    # @__apply_fun_to_coordinates results in FUN(f(x0),g(x0)),FUN(f(x1),g(x1)),...,FUN(f(xn),g(xn))
    # Implementation is defined on family of functions from (REACTIVE_UNIT_DIST[.]).
    def __apply_fun_to_coordinates(self, FUN, calculator: Calculator):
        return FUN([weight * calculator.pdf(ru) for weight, ru in
                    zip(self._weights, self._reactive_units)])

    # def show(self):
    #     DOMAIN = inmem['DOMAIN']
    #     plt.plot(DOMAIN, self.discretized_distribution(), 'o', DOMAIN, self.discretized_distribution(), '--')
    #     plt.legend(['data', 'cubic'], loc='best')
    #     plt.show()

    def deactivate(self):
        self.is_active = False


@dataclasses.dataclass
class NewWord:
    word_id: int
    active = True

    def __hash__(self):
        return self.word_id

    def deactivate(self):
        self.active = False


@dataclasses.dataclass
class GameParams:
    population_size: int
    steps: int
    stimulus: str
    max_num: int
    runs: int
    guessing_game_2: bool
    seed: int
    discriminative_threshold: float
    discriminative_history_length: int
    delta_inc: float
    delta_dec: float
    delta_inh: float
    alpha: float
    beta: float
    super_alpha: float


class ConnectionMatrixLxC:
    def __init__(self, max_row: int, max_col: int):
        self._square_matrix = np.zeros((max_row, max_col))
        self._row = 0
        self._col = 0

    @staticmethod
    def new_matrix(steps: int):
        row = int(3*np.sqrt(steps))
        col = int(3*np.sqrt(steps))

        return ConnectionMatrixLxC(row, col)

    def __call__(self, row, col) -> float:
        return self._square_matrix[row, col]

    def to_ndarray(self) -> ndarray:
        return self._square_matrix

    def rows(self):
        return self._square_matrix.shape[0]

    def cols(self):
        return self._square_matrix.shape[1]

    def get_rows_all_smaller_than_threshold(self, threshold: float) -> ndarray:
        result, = np.where(np.all(self._square_matrix < threshold, axis=1))
        return result

    def get_row_argmax(self, row_index) -> int:
        return np.argmax(self._square_matrix, axis=1)[row_index]

    def get_col_argmax(self, col_index) -> int:
        return np.argmax(self._square_matrix, axis=0)[col_index]

    def update_cell(self, row: int, column: int, update: Callable[[float], float]):
        recomputed_value = update(self._square_matrix[row, column])
        self._square_matrix[row, column] = recomputed_value

    def update_matrix_on_given_row(self, row_index: int, column_indices: (List[int] or ndarray),
                                   scalar: float):
        updated_cells = self._square_matrix[row_index, column_indices]
        self._square_matrix[row_index, column_indices] += scalar * updated_cells

    def reset_matrix_on_row_indices(self, row_indices: (List[int] or ndarray)):
        self._square_matrix[row_indices, :] = 0

    def reset_matrix_on_col_indices(self, col_indices: (List[int] or ndarray)):
        self._square_matrix[:, col_indices] = 0

    def reduce(self, height: int, width: int):
        return self._square_matrix[:height, :width]

    def get_row_vector(self, word_index) -> ndarray:
        return self._square_matrix[word_index, :]

    def add_new_row(self):
        self._row += 1
        if self._row == self._square_matrix.shape[0]:
            self._double_height()

    def add_new_col(self):
        self._col += 1
        if self._col == self._square_matrix.shape[1]:
            self._double_width()

    def _double_height(self):
        height, width = self._square_matrix.shape
        new_m = np.zeros((2 * height, width))
        new_m[0:height, :] = self._square_matrix
        self._square_matrix = new_m

    def _double_width(self):
        height, width = self._square_matrix.shape
        new_m = np.zeros((height, width * 2))
        new_m[:, 0:width] = self._square_matrix
        self._square_matrix = new_m


class NewAgent:
    def __init__(self, agent_id: int, game_params: GameParams):
        self.agent_id = agent_id
        self._game_params = game_params
        self._lxc = ConnectionMatrixLxC.new_matrix(game_params.steps)
        self._lexicon: List[NewWord] = []
        self._categories: List[NewCategory] = []
        self._lex2index = {}
        self._discriminative_success = []
        self._communicative_success1 = []
        self._communicative_success2 = []
        self._discriminative_success_mean = 0.

    def __repr__(self):
        actual_lexicon = len(self._lexicon)
        actual_categories = len(self._categories)
        return f'{self.agent_id} {str(self._lxc.reduce(actual_lexicon, actual_categories))}'

    @staticmethod
    def to_dict(agent) -> Dict:
        words = [{'word_id': w.word_id, 'word_position': agent._lex2index[w], 'active': w.active}
                 for w in agent._lexicon]

        categories = [{'category_id': category.category_id,
                       'is_active': category.is_active,
                       'reactive_units': [r if isinstance(r, int) else [*r] for r in category.reactive_units()],
                       'weights': category.weights()} for category in agent._categories]

        discriminative_success = list(agent._discriminative_success)

        lxc = agent._lxc.reduce(len(words), len(categories)).tolist()

        return {'agent_id': agent.agent_id,
                'categories': categories,
                'words': words,
                'discriminative_success': discriminative_success,
                'lxc': lxc}

    def get_most_connected_word(self, category: NewCategory, activation_threshold=0) -> Union[NewWord, None]:
        category_index = category.category_id
        category_argmax = self._lxc.get_col_argmax(category_index)
        value = self._lxc(category_argmax, category_index)
        if value > activation_threshold and category_argmax < len(self._lexicon):
            return self._lexicon[category_argmax]
        else:
            return None

    def get_most_connected_category(self, word: NewWord, activation_threshold=0) -> Union[NewCategory, None]:
        word_index = self._lex2index[word]
        word_argmax = self._lxc.get_row_argmax(word_index)
        value = self._lxc(word_index, word_argmax)
        if value > activation_threshold:
            return self._categories[word_argmax]
        else:
            return None

    def has_categories(self) -> bool:
        return len(self._categories) > 0

    def get_best_matching_category(self, stimulus, calculator: Calculator) -> NewCategory:
        responses = [c.response(stimulus, calculator) for c in self._categories if c.is_active]
        response_argmax = np.argmax(responses)
        return self._categories[response_argmax]

    def knows_word(self, w: NewWord):
        return w in self._lex2index.keys()

    def forget_categories(self, category_in_use: NewCategory):
        active_categories = [c for c in self._categories if c.is_active]

        for c in active_categories:
            c.decrement_weights(self._game_params.alpha)

        to_forget = [i for i, activate_category in enumerate(active_categories) if
                     activate_category.max_weigth() < self._game_params.super_alpha and category_in_use != i]

        self._lxc.reset_matrix_on_col_indices(to_forget)
        [self._categories[c].deactivate() for c in to_forget]

    def forget_words(self, super_alpha=.01):
        to_forget = self._lxc.get_rows_all_smaller_than_threshold(super_alpha)
        to_forget = [i for i in to_forget if i < len(self._lexicon)]
        self._lxc.reset_matrix_on_row_indices(to_forget)
        [self._lexicon[i].deactivate() for i in to_forget]

    def add_new_word(self, w: NewWord):
        self._lex2index[w] = len(self._lexicon)
        self._lexicon.append(w)
        self._lxc.add_new_row()

    def inhibit_word2categories_connections(self, word: NewWord, except_category: NewCategory):
        word_index = self._lex2index[word]
        # except_category_index = self._cat2index[except_category]
        indices = [i for i in range(len(self._categories)) if i != except_category.category_id]
        self._lxc.update_matrix_on_given_row(word_index, indices, -self._game_params.delta_inh)

    def learn_word_category(self, word: NewWord, category: NewCategory, connection=.5):
        self.__update_connection(word, category, lambda v: connection)

    def update_on_success(self, word: NewWord, category: NewCategory):
        self.__update_connection(word, category, lambda v: v + self._game_params.delta_inc * v)

    def update_on_failure(self, word: NewWord, category: NewCategory):
        self.__update_connection(word, category, lambda v: v - self._game_params.delta_dec * v)

    def learn_stimulus(self, stimulus: NewAbstractStimulus, calculator: Calculator):
        if self._discriminative_success_mean >= self._game_params.discriminative_threshold:
            logger.debug("updating category by adding reactive unit centered on %s" % stimulus)
            category = self.get_best_matching_category(stimulus, calculator)
            logger.debug("updating category")
            category.add_reactive_unit(stimulus)
        else:
            logger.debug(f'adding new category centered on {stimulus}')
            self.add_new_category(stimulus)

    def add_new_category(self, stimulus: NewAbstractStimulus, weight=0.5):
        category_id = len(self._categories)
        new_category = NewCategory(category_id=category_id)
        new_category.add_reactive_unit(stimulus, weight)
        self._categories.append(new_category)
        self._lxc.add_new_col()
        # assert (len(self._categories) <= self._lxc.cols()), \
        #     'categories size must be at most the size of the lxc matrix height'

    def reinforce_category(self, category: NewCategory, stimulus, calculator: Calculator):
        category.reinforce(stimulus, self._game_params.beta, calculator)

    def __update_connection(self, word: NewWord, category: NewCategory, update: Callable[[float], float]):
        word_index = self._lex2index[word]
        self._lxc.update_cell(word_index, category.category_id, update)

    def csimilarity(self, word: NewWord, category: NewCategory, calculator: Calculator):
        area = category.union(calculator)
        # omit multiplication by x_delta because all we need is ratio: coverage/area:
        word_meaning = self.word_meaning(word, calculator)
        coverage = np.minimum(word_meaning, area)

        return sum(coverage) / sum(area)

    # based on how much the word meaning covers the category
    def word_meaning(self, word: NewWord, calculator: Calculator) -> float:
        word_index = self._lex2index[word]
        word2categories_vector = self._lxc.get_row_vector(word_index)[:len(self._lexicon)]
        return np.dot([c.union(calculator) for c in self._categories], word2categories_vector)
        # return sum([category.union() * word2category_weight for category, word2category_weight in
        #             zip(self._categories, self._lxc.get_row_vector(word_index))])

    def semantic_meaning(self, word: NewWord, stimuli: NewAbstractStimulus):
        word_index = self._lex2index[word]

        activations = [
            sum([float(c.response(s) > 0.0) * float(self._lxc(word_index, c.category_id) > 0.0)
                 for c in self._categories]) for s in stimuli]

        flat_bool_activations = list(map(lambda x: int(x > 0.0), activations))
        mean_bool_activations = []
        for i in range(0, len(flat_bool_activations)):
            window = flat_bool_activations[max(0, i - 5):min(len(flat_bool_activations), i + 5)]
            mean_bool_activations.append(int(sum(window) / len(window) > 0.5))

        return mean_bool_activations if self.stm == 'quotient' else flat_bool_activations

    def is_monotone(self, word: NewWord, stimuli):
        bool_activations = self.semantic_meaning(word, stimuli)
        alt = len([a for a, aa in zip(bool_activations, bool_activations[1:]) if a != aa])
        return alt == 1

    def add_discrimination_success(self, history=50):
        self._discriminative_success.append(True)
        self._discriminative_success_mean = np.mean(self._discriminative_success[-history:])

    def add_discriminative_failure(self, history=50):
        self._discriminative_success.append(False)
        self._discriminative_success_mean = np.mean(self._discriminative_success[-history:])

    def get_communicative_success1(self):
        return self._communicative_success1

    def get_communicative_success2(self):
        return self._communicative_success2

    def add_communicative1_success(self):
        self._communicative_success1.append(True)

    def add_communicative1_failure(self):
        self._communicative_success1.append(False)

    def add_communicative2_success(self):
        self._communicative_success2.append(True)

    def add_communicative2_failure(self):
        self._communicative_success2.append(False)


class ThreadSafeWordFactory:

    def __init__(self) -> None:
        self._counter = 0
        self._lock = Lock()

    def __call__(self) -> NewWord:
        with self._lock:
            w = NewWord(self._counter)
            self._counter += 1
            return w


class AggregatedGameResultStats:
    def __init__(self, game_params: GameParams) -> None:
        self._agent2discrimination_success = {}
        self._agent2communication_success1 = {}
        self._agent2communication_success2 = {}
        self._agent2communication_success12 = {}
        self._game_params = game_params

    def add_discrimination_success(self, agent):
        self._add_discrimination_result(self._agent2discrimination_success, agent, True)

    def add_discrimination_failure(self, agent):
        self._add_discrimination_result(self._agent2discrimination_success, agent, False)

    def add_communication1_success(self, agent):
        self._add_discrimination_result(self._agent2communication_success1, agent, True)

    def add_communication1_failure(self, agent):
        self._add_discrimination_result(self._agent2communication_success1, agent, False)

    def add_communication2_success(self, agent):
        self._add_discrimination_result(self._agent2communication_success2, agent, True)

    def add_communication12_failure(self, agent):
        self._add_discrimination_result(self._agent2communication_success12, agent, False)

    def add_communication12_success(self, agent):
        self._add_discrimination_result(self._agent2communication_success12, agent, True)

    def add_communication2_failure(self, agent):
        self._add_discrimination_result(self._agent2communication_success2, agent, False)

    def _add_discrimination_result(self, agent2dict: Dict[int, deque], agent, result: bool):
        if agent.agent_id not in agent2dict.keys():
            agent2dict[agent.agent_id] = deque(maxlen=self._game_params.discriminative_history_length)

        agent2dict[agent.agent_id].append(result)
