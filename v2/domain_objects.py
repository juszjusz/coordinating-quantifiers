import dataclasses
import logging
from threading import Lock
from typing import Tuple, Callable, List, Dict, Union

import numpy as np
from numpy import ndarray

from calculator import Calculator, NewAbstractStimulus, StimulusContext, Stimulus

logger = logging.getLogger(__name__)


class NewCategory:
    def __init__(self, category_id: int):
        self.category_id = category_id
        self._weights = []
        self._reactive_units = []

    @staticmethod
    def make_copy(c):
        category = NewCategory(c.category_id)
        category._weights = c._weights.copy()
        category._reactive_units = c._reactive_units.copy()
        return category

    def __hash__(self):
        return self.category_id

    def __repr__(self):
        return f'[{self._weights}x{self._reactive_units}]'

    def reactive_units(self):
        return self._reactive_units

    def weights(self):
        return self._weights

    def response(self, stimulus: Stimulus, calculator: Calculator):
        return sum([weight * calculator.dot_product(ru_value, stimulus) for weight, ru_value in
                    zip(self._weights, self._reactive_units)])

    def add_reactive_unit(self, stimulus: Stimulus, weight=0.5):
        self._weights.append(weight)
        self._reactive_units.append(stimulus)

    def select(self, context: StimulusContext, calculator: Calculator) -> int or None:
        s1, s2 = context
        r1, r2 = self.response(s1, calculator), self.response(s2, calculator)
        if r1 == r2:
            return None
        else:
            return np.argmax([r1, r2])

    def reinforce(self, stimulus: Stimulus, beta, calculator: Calculator):
        self._weights = [weight + beta * calculator.dot_product(ru, stimulus) for weight, ru in
                         zip(self._weights, self._reactive_units)]

    def decrement_weights(self, alpha):
        self._weights = [weight - alpha * weight for weight in self._weights]

    def max_weight(self):
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


@dataclasses.dataclass
class NewWord:
    word_id: int
    originated_from_category: NewCategory

    def __hash__(self):
        return self.word_id


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
        row = int(3 * np.sqrt(steps))
        col = int(3 * np.sqrt(steps))

        return ConnectionMatrixLxC(row, col)

    def __call__(self, row: int, col: int) -> float:
        return self._square_matrix[row, col]

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

    def update_matrix_on_given_row(self, row_index: int, scalar: float):
        updated_cells = self._square_matrix[row_index, :]
        self._square_matrix[row_index, :] += scalar * updated_cells

    def reset_matrix_on_row_indices(self, row_indices: Union[List[int], ndarray]):
        self._square_matrix[row_indices, :] = 0

    def reset_matrix_on_col_indices(self, col_indices: Union[List[int], ndarray]):
        self._square_matrix[:, col_indices] = 0

    def reduce(self):
        return self._square_matrix[:self._row, :self._col]

    def get_row_vector(self, word_index) -> ndarray:
        row_vector = self._square_matrix[word_index, :]
        adjusted_row_vector = row_vector[:self._col]
        return adjusted_row_vector

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
        self._lexicon: List[Tuple[NewWord, bool]] = []
        self._categories: List[Tuple[NewCategory, bool]] = []
        self._lex2index = {}
        self._cat2index = {}
        self._discriminative_success = [False]
        self._communicative_success1 = [False]
        self._communicative_success2 = [False]
        self._discriminative_success_means = [0.]
        self._active_lexicon_size_history = [0]

    def __repr__(self):
        return f'{self.agent_id} {str(self._lxc.reduce())}'

    @staticmethod
    def to_dict(agent) -> Dict:
        words = [{'word_id': w.word_id, 'word_position': agent._lex2index[w], 'active': active}
                 for w, active in agent._lexicon]

        categories = [{'category_id': category.category_id,
                       'is_active': active,
                       'reactive_units': [r if isinstance(r, int) else [*r] for r in category.reactive_units()],
                       'weights': category.weights()} for (category, active) in agent._categories]

        discriminative_success = list(agent._discriminative_success)

        lxc = agent._lxc.reduce().tolist()

        return {'agent_id': agent.agent_id,
                'categories': categories,
                'words': words,
                'discriminative_success': discriminative_success,
                'communicative_success1': agent._communicative_success1,
                'lxc': lxc}

    def get_most_connected_word(self, category: NewCategory, activation_threshold=0) -> Union[NewWord, None]:
        category_index = self._cat2index[category]
        word_category_maximizer = self._lxc.get_col_argmax(category_index)
        wXc_value = self._lxc(word_category_maximizer, category_index)
        if wXc_value > activation_threshold:
            word, active = self._lexicon[word_category_maximizer]
            assert active, 'WxC > activation_th -> word active'
            return word
        else:
            return None

    def get_most_connected_category(self, word: NewWord, activation_threshold=0) -> Union[NewCategory, None]:
        word_index = self._lex2index[word]

        _, active = self._lexicon[word_index]
        assert active, 'only active words'

        category_word_maximizer = self._lxc.get_row_argmax(word_index)
        wXc_value = self._lxc(word_index, category_word_maximizer)
        if wXc_value > activation_threshold:
            c, active = self._categories[category_word_maximizer]
            assert active, 'WxC > activation_th -> category active'
            return c
        else:
            return None

    def get_categories(self) -> List[Tuple[NewCategory, bool]]:
        return self._categories

    def get_active_categories(self) -> List[NewCategory]:
        return [c for c, active in self._categories if active]

    def get_active_words(self) -> List[NewWord]:
        return [w for w, active in self._lexicon if active]

    def has_categories(self) -> bool:
        return len(self.get_active_categories()) > 0

    def get_best_matching_category(self, stimulus, calculator: Calculator) -> NewCategory:
        active_categories = self.get_active_categories()
        responses = [c.response(stimulus, calculator) for c in active_categories]
        response_argmax = np.argmax(responses)
        return active_categories[response_argmax]

    def knows_word(self, w: NewWord):
        active_words = self.get_active_words()
        return w in active_words

    def forget_categories(self, category_in_use: NewCategory):
        active_categories = self.get_active_categories()

        for c in active_categories:
            c.decrement_weights(self._game_params.alpha)

        to_forget = [self._cat2index[activate_category] for activate_category in active_categories if
                     activate_category.max_weight() < self._game_params.super_alpha and category_in_use != activate_category]

        self._lxc.reset_matrix_on_col_indices(to_forget)
        for i in to_forget:
            c, _ = self._categories[i]
            self._categories[i] = (c, False)

    def forget_words(self, super_alpha=.01):
        to_forget = self._lxc.get_rows_all_smaller_than_threshold(super_alpha)
        to_forget = [i for i in to_forget if i < len(self._lexicon)]
        self._lxc.reset_matrix_on_row_indices(to_forget)
        for i in to_forget:
            w, _ = self._lexicon[i]
            self._lexicon[i] = (w, False)

    def update_discriminative_success_mean(self, history=50):
        discriminative_success_mean = np.mean(self._discriminative_success[-history:])
        self._discriminative_success_means.append(discriminative_success_mean)

    def add_new_word(self, w: NewWord):
        self._lex2index[w] = len(self._lexicon)
        self._lexicon.append((w, True))
        self._lxc.add_new_row()

    def add_new_category(self, stimulus: NewAbstractStimulus, weight=0.5):
        category_index = len(self._categories)
        new_category = NewCategory(category_id=category_index)
        self._cat2index[new_category] = category_index
        new_category.add_reactive_unit(stimulus, weight)
        self._categories.append((new_category, True))
        self._lxc.add_new_col()

    def inhibit_word2categories_connections(self, word: NewWord, except_category: NewCategory):
        word_index = self._lex2index[word]
        category_index = self._cat2index[except_category]
        retained_value = self._lxc(word_index, category_index)
        self._lxc.update_matrix_on_given_row(word_index, -self._game_params.delta_inh)
        self._lxc.update_cell(word_index, category_index, lambda v: retained_value)

    def learn_word_category(self, word: NewWord, category: NewCategory, connection=.5):
        self.__update_connection(word, category, lambda v: connection)

    def update_on_success(self, word: NewWord, category: NewCategory):
        self.__update_connection(word, category, lambda v: v + self._game_params.delta_dec * v)
        self.inhibit_word2categories_connections(word=word, except_category=category)

    def update_on_failure(self, word: NewWord, category: NewCategory):
        self.__update_connection(word, category, lambda v: v - self._game_params.delta_dec * v)

    def learn_stimulus(self, stimulus: NewAbstractStimulus, calculator: Calculator):
        if self._discriminative_success_means[-1] >= self._game_params.discriminative_threshold:
            logger.debug("updating category by adding reactive unit centered on %s" % str(stimulus))
            category = self.get_best_matching_category(stimulus, calculator)
            logger.debug("updating category")
            category.add_reactive_unit(stimulus)
        else:
            logger.debug(f'adding new category centered on {stimulus}')
            self.add_new_category(stimulus)

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

        # based on how much the word meaning covers the category
        return sum(coverage) / sum(area)


    def get_word_meanings(self, calculator: Calculator) -> Dict[NewWord, List[NewAbstractStimulus]]:
        words = self.get_active_words()
        meanings = {}
        for word in words:
            meanings[word] = self.word_meaning_new(word, calculator.values(), calculator)
        return meanings

    def word_meaning_new(self, word: NewWord, stimuli: List, calculator: Calculator):
        # [f] = {q : SUM L(f,c)*<c|R_q> > 0} <=> [f] = {q : L(f,c) > 0 & <c|R_q> > 0}
        word_index = self._lex2index[word]
        word2categories_vector = self._lxc.get_row_vector(word_index)
        non_zero_cats = np.nonzero(word2categories_vector)[0]
        # cs = [c for i in non_zero_cats for c, active in self._categories[i] if active]

        cs = [self._categories[i][0] for i in non_zero_cats]
        return [q for q in stimuli for c in cs if c.response(q, calculator)]

    def word_meaning(self, word: NewWord, calculator: Calculator) -> float:
        active_categories = self.get_active_categories()
        word_index = self._lex2index[word]
        word2categories_vector = self._lxc.get_row_vector(word_index)[:len(self._lexicon)]
        return np.dot([c.union(calculator) for c in active_categories], word2categories_vector)
        # return sum([category.union() * word2category_weight for category, word2category_weight in
        #             zip(self._categories, self._lxc.get_row_vector(word_index))])

    def semantic_meaning(self, word: NewWord, stimuli: List[NewAbstractStimulus], calculator: Calculator):
        word_index = self._lex2index[word]

        activations = [
            sum([float(c.response(s, calculator) > 0.0) * float(self._lxc(word_index, c.category_id) > 0.0)
                 for c, active in self._categories if active]) for s in stimuli]

        flat_bool_activations = list(map(lambda x: int(x > 0.0), activations))
        mean_bool_activations = []
        for i in range(0, len(flat_bool_activations)):
            window = flat_bool_activations[max(0, i - 5):min(len(flat_bool_activations), i + 5)]
            mean_bool_activations.append(int(sum(window) / len(window) > 0.5))

        return mean_bool_activations
        # return mean_bool_activations if self.stm == 'quotient' else flat_bool_activations

    def get_monotonicity(self, stimuli: List[NewAbstractStimulus], calculator: Calculator):
        active_lexicon = [w for w, active in self._lexicon if active]
        mons = [self.is_monotone(w, stimuli, calculator) for w in active_lexicon]
        return mons.count(True) / len(mons) if len(mons) > 0 else 0.0

    def is_monotone(self, word: NewWord, stimuli, calculator: Calculator):
        bool_activations = self.semantic_meaning(word, stimuli, calculator)
        alt = len([a for a, aa in zip(bool_activations, bool_activations[1:]) if a != aa])
        return alt == 1

    def get_discriminative_success(self):
        return self._discriminative_success_means

    def add_discrimination_success(self):
        self._discriminative_success.append(True)

    def add_discriminative_failure(self):
        self._discriminative_success.append(False)

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

    def __call__(self, originated_from: NewCategory) -> NewWord:
        with self._lock:
            w = NewWord(self._counter, originated_from)
            self._counter += 1
            return w
