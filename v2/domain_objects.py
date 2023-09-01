import bisect
import dataclasses
import logging
from copy import copy
from typing import Callable, List, Dict, Union

import numpy as np
from tqdm import tqdm

from calculator import Calculator, NewAbstractStimulus, StimulusContext, Stimulus
from matrix_datastructure import Matrix, One2OneMapping

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


class NewCategory:
    def __init__(self, category_id: int):
        self.category_id = category_id
        self._weights = []
        self._reactive_units = []

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.category_id == other.category_id

    def __copy__(self):
        category = NewCategory(self.category_id)
        category._weights = self._weights.copy()
        category._reactive_units = self._reactive_units.copy()
        return category

    def __hash__(self):
        return self.category_id

    def __repr__(self):
        return f'{self.category_id}[{self._weights}x{self._reactive_units}]'

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

    def select(self, context: StimulusContext, calculator: Calculator) -> Union[int, None]:
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

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.word_id == other.word_id

    def __copy__(self):
        return NewWord(self.word_id, copy(self.originated_from_category))


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


def register_agent_update_operation(update):
    def update_call(agent, *args, **kwargs):
        if update.__name__ == 'next_step':
            agent.updates_history.append([])

        current_step = agent.updates_history[-1]
        args_copy = [copy(a) for a in args]
        current_step.append([update.__name__, args_copy, kwargs])
        update(agent, *args, **kwargs)

    return update_call


class NewAgent:
    def __init__(self, agent_id: int, calculator: Calculator, game_params: GameParams):
        self.agent_id = agent_id
        self.updates_history = []
        self._game_params = game_params
        self._lxc = LxC.new_matrix(game_params.steps)
        self._discriminative_success = [False]
        self._communicative_success1 = [False]
        self._communicative_success2 = [False]
        self._discriminative_success_means = [0.]
        self._new_category_counter = SimpleCounter()
        self._calculator = calculator

    def __repr__(self):
        return f'{self.agent_id}'

    @staticmethod
    def snapshot(agent):
        snapshot = NewAgent(agent_id=agent.agent_id, calculator=agent._calculator, game_params=agent._game_params)
        snapshot._lxc = copy(agent._lxc)
        return snapshot

    @staticmethod
    def recreate_from_history(agent_id: int, calculator: Calculator, game_params: GameParams, updates_history: List,
                              max_step: int = -1, snapshot_rate: int = 10):
        snapshots = []
        agent = NewAgent(agent_id=agent_id, calculator=calculator, game_params=game_params)

        if max_step > len(updates_history):
            logger.warning('can recreate at most ' + str(len(updates_history)) + ' fall back to full history')
        if max_step > 0:
            updates_history = updates_history[:max_step]

        for step, step_updates in enumerate(tqdm(updates_history, f'recreating agent {agent_id} by updates')):
            for method_name, args, kwargs in step_updates:
                msg = f'meth: {method_name} {args}'
                logger.debug(msg)

                agent_method = getattr(agent, method_name)
                agent_method(*args, **kwargs)

            if step % snapshot_rate == 0:
                snapshots.append((step_updates, NewAgent.snapshot(agent)))

        return snapshots

    @staticmethod
    def to_dict(agent) -> Dict:
        agent._lxc.remove_nonactive_categories()
        agent._lxc.remove_nonactive_words()

        words = [{'word_id': w.word_id, 'originated_from_category': {
            'reactive_units': [r if isinstance(r, int) else [*r] for r in w.originated_from_category.reactive_units()],
            'weights': [round(w, 3) for w in w.originated_from_category.weights()]
        }}
                 for w in agent.get_words()]

        categories = [{'category_id': category.category_id,
                       'reactive_units': [r if isinstance(r, int) else [*r] for r in category.reactive_units()],
                       'weights': [round(w, 3) for w in category.weights()]} for category in
                      agent.get_categories()]

        discriminative_success = list(agent._discriminative_success)

        return {'agent_id': agent.agent_id,
                'categories': categories,
                'words': words,
                'discriminative_success': discriminative_success,
                'communicative_success1': agent._communicative_success1,
                'lxc': agent._lxc.get_matrix().tolist()}

    def has_categories(self) -> bool:
        return len(self._lxc.get_active_categories()) > 0

    def get_words(self) -> List[NewWord]:
        return self._lxc.get_active_words()

    def get_categories(self) -> List[NewCategory]:
        return self._lxc.get_active_categories()

    def get_best_matching_category(self, stimulus) -> NewCategory:
        active_categories = self._lxc.get_active_categories()
        responses = [c.response(stimulus, self._calculator) for c in active_categories]
        response_argmax = np.argmax(responses)
        return active_categories[response_argmax]

    def get_most_connected_word(self, category: NewCategory, activation_threshold=0) -> Union[NewWord, None]:
        return self._lxc.get_most_connected_word(category, activation_threshold)

    def get_most_connected_category(self, word: NewWord, activation_threshold=0) -> Union[NewCategory, None]:
        return self._lxc.get_most_connected_category(word, activation_threshold)

    def knows_word(self, w: NewWord):
        active_words = self._lxc.get_active_words()
        return w in active_words

    @register_agent_update_operation
    def forget_categories(self, category_in_use: NewCategory):
        self._lxc.forget_categories(category_in_use, self._game_params.alpha, self._game_params.super_alpha)

    @register_agent_update_operation
    def forget_words(self, super_alpha=.01):
        # todo use super_alpha from game parrams
        self._lxc.forget_words(super_alpha)

    @register_agent_update_operation
    def update_discriminative_success_mean(self, history=50):
        discriminative_success_mean = np.mean(self._discriminative_success[-history:])
        self._discriminative_success_means.append(discriminative_success_mean)

    @register_agent_update_operation
    def add_new_word(self, w: NewWord):
        self._lxc.add_new_or_reactivate_word(w)

    @register_agent_update_operation
    def learn_word_category(self, word: NewWord, category: NewCategory, connection=.5):
        self._lxc.update_word_category_connection(word, category, lambda v: connection)

    @register_agent_update_operation
    def update_on_success(self, word: NewWord, category: NewCategory):
        self._lxc.update_word_category_connection(word, category, lambda v: v + self._game_params.delta_dec * v)
        self._inhibit_word2categories_connections(word=word, except_category=category)

    def _inhibit_word2categories_connections(self, word: NewWord, except_category: NewCategory):
        retained_value = self._lxc.get_connection(word, except_category)
        self._lxc.update_row_connection(word, scalar=-self._game_params.delta_inh)
        self._lxc.update_word_category_connection(word, except_category, lambda v: retained_value)

    @register_agent_update_operation
    def update_on_failure(self, word: NewWord, category: NewCategory):
        self._lxc.update_word_category_connection(word, category, lambda v: v - self._game_params.delta_dec * v)

    @register_agent_update_operation
    def learn_stimulus(self, stimulus: Stimulus, weight=.5):
        if self._discriminative_success_means[-1] >= self._game_params.discriminative_threshold:
            logger.debug("updating category by adding reactive unit centered on %s" % str(stimulus))
            category = self.get_best_matching_category(stimulus)
            logger.debug("updating category")
            category.add_reactive_unit(stimulus)
        else:
            logger.debug(f'adding new category centered on {stimulus}')
            category_id = self._new_category_counter()
            new_category = NewCategory(category_id=category_id)
            new_category.add_reactive_unit(stimulus, weight)
            self._lxc.add_new_category(new_category)

    @register_agent_update_operation
    def reinforce_category(self, category: NewCategory, stimulus):
        # retain previously created category reference, crucial for correct agent reconstruction
        stored_category = self._lxc.get_stored_category(category)
        stored_category.reinforce(stimulus, self._game_params.beta, self._calculator)

    @register_agent_update_operation
    def add_discrimination_success(self):
        self._discriminative_success.append(True)

    @register_agent_update_operation
    def add_discriminative_failure(self):
        self._discriminative_success.append(False)

    @register_agent_update_operation
    def add_communicative1_success(self):
        self._communicative_success1.append(True)

    @register_agent_update_operation
    def add_communicative1_failure(self):
        self._communicative_success1.append(False)

    @register_agent_update_operation
    def add_communicative2_success(self):
        self._communicative_success2.append(True)

    @register_agent_update_operation
    def add_communicative2_failure(self):
        self._communicative_success2.append(False)

    @register_agent_update_operation
    def next_step(self):
        # mark end of the game between agents
        pass

    def select_stimuli_by_category(self, category: NewCategory, context: StimulusContext) -> Stimulus:
        return category.select(context, self._calculator)

    def csimilarity(self, word: NewWord, category: NewCategory):
        area = category.union(self._calculator)
        # omit multiplication by x_delta because all we need is ratio: coverage/area:
        word_meaning = self.word_meaning(word, self._calculator)
        coverage = np.minimum(word_meaning, area)

        # based on how much the word meaning covers the category
        return sum(coverage) / sum(area)

    def compute_word_meanings(self) -> Dict[NewWord, List[Stimulus]]:
        return self._lxc.word_meanings(self._calculator.values(), self._calculator)

    @staticmethod
    def is_monotone_new(word_meaning: List[Stimulus], calculator: Calculator):

        lower = min(word_meaning, key=lambda x: float(x[0]/x[1]))
        upper = max(word_meaning, key=lambda x: float(x[0]/x[1]))
        meanings_space = calculator.values()
        start_index = bisect.bisect(meanings_space, lower)
        upper_index = bisect.bisect(meanings_space, upper)
        if start_index == 0 and meanings_space[0:upper_index] == word_meaning:
            return True
        elif upper_index == len(meanings_space) and meanings_space[upper_index:] == word_meaning:
            return True
        return False

    def word_meaning(self, word: NewWord) -> float:
        active_categories = self.get_active_categories()
        word_index = self._lex2index[word]
        word2categories_vector = self._lxc.get_row_vector(word_index)[:len(self._lexicon)]
        return np.dot([c.union(self._calculator) for c in active_categories], word2categories_vector)
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

    def get_communicative_success1(self):
        return self._communicative_success1

    def get_communicative_success2(self):
        return self._communicative_success2


class SimpleCounter:
    def __init__(self) -> None:
        self._counter = 0

    def __call__(self) -> int:
        i = self._counter
        self._counter += 1
        return i


class LxC:
    def __init__(self, row: int, col: int):
        self._categories = One2OneMapping({}, {})
        self._words = One2OneMapping({}, {})
        self._lxc = Matrix(row, col)

    def __copy__(self):
        self.remove_nonactive_categories()
        self.remove_nonactive_words()

        lxc = LxC(0, 0)
        lxc._lxc = copy(self._lxc)
        lxc._categories = copy(self._categories)
        lxc._words = copy(self._words)
        return lxc

    @staticmethod
    def new_matrix(steps: int):
        row = int(3 * np.sqrt(steps))
        col = int(3 * np.sqrt(steps))

        return LxC(row, col)

    def get_stored_category(self, c: NewCategory) -> NewCategory:
        return self._categories.get_stored_object(c)

    def get_matrix(self) -> np.ndarray:
        return self._lxc.reduce()

    def add_new_or_reactivate_word(self, w: NewWord):
        if w in self._words.nonactive_elements():
            # activate previously deactivated word
            word_index = self._words.get_object_index(w)
            self._words.reactivate_element_at_index(word_index)
        else:
            # add new word
            self.remove_nonactive_words()
            self._words.add_new_element(w)
            self._lxc.add_new_row()

    def add_new_category(self, new_category: NewCategory):
        self._categories.add_new_element(new_category)
        self._lxc.add_new_col()

    def get_most_connected_word(self, category: NewCategory, activation_threshold=0) -> Union[NewWord, None]:
        category_index = self._categories.get_object_index(category)
        word_category_maximizer_index = self._lxc.get_col_argmax(category_index)
        wXc_value = self._lxc(word_category_maximizer_index, category_index)
        if wXc_value > activation_threshold:
            word_category_maximizer, active = self._words.get_object_by_index(word_category_maximizer_index)
            assert active, 'WxC > activation_th -> word active'
            return word_category_maximizer
        else:
            return None

    def get_most_connected_category(self, word: NewWord, activation_threshold) -> Union[NewCategory, None]:
        word_index = self._words.get_object_index(word)

        _, active = self._words.get_object_by_index(word_index)
        assert active, 'only active words'

        category_word_maximizer_index = self._lxc.get_row_argmax(word_index)
        wXc_value = self._lxc(word_index, category_word_maximizer_index)
        if wXc_value > activation_threshold:
            category_word_maximizer, active = self._categories.get_object_by_index(category_word_maximizer_index)
            assert active, 'WxC > activation_th -> category active'
            return category_word_maximizer
        else:
            return None

    def get_active_categories(self) -> List[NewCategory]:
        return self._categories.active_elements()

    def get_active_words(self) -> List[NewWord]:
        return self._words.active_elements()

    def forget_categories(self, category_in_use: NewCategory, alpha: float, super_alpha: float):
        active_categories = self._categories.active_elements()

        for c in active_categories:
            c.decrement_weights(alpha)

        to_forget = [activate_category for activate_category in active_categories if
                     activate_category.max_weight() < super_alpha and category_in_use != activate_category]

        to_forget_indices = [self._categories.get_object_index(c) for c in to_forget]
        self._lxc.reset_matrix_on_col_indices(to_forget_indices)
        [self._categories.deactivate_element(c) for c in to_forget]

        if self._categories.sparsity_rate() > .4:
            self.remove_nonactive_categories()

    def remove_nonactive_categories(self):
        to_remove = self._categories.nonactive_elements()
        to_remove_indices = [self._categories.get_object_index(c) for c in to_remove]
        self._lxc.remove_columns(to_remove_indices)
        self._categories.remove_nonactive_and_reindex()
        assert self._lxc.cols() == len(
            self._categories), 'after deletion column vectors correspond to categories in 1-1 manner'

    def forget_words(self, super_alpha: float):
        to_forget_indices = self._lxc.get_rows_all_smaller_than_threshold(super_alpha)
        self._lxc.reset_matrix_on_row_indices(to_forget_indices)
        to_forget = [self._words.get_object_by_index(i) for i in to_forget_indices]

        [self._words.deactivate_element(w) for w, _ in to_forget]

        if self._words.sparsity_rate() > .4:
            self.remove_nonactive_words()

    def remove_nonactive_words(self):
        to_remove = self._words.nonactive_elements()
        to_remove_indices = [self._words.get_object_index(w) for w in to_remove]
        self._lxc.remove_rows(to_remove_indices)
        self._words.remove_nonactive_and_reindex()
        assert self._lxc.rows() == len(
            self._words), 'after deletion row vectors correspond to words in 1-1 manner'

    def update_row_connection(self, word: NewWord, scalar):
        word_index = self._words.get_object_index(word)
        _, active = self._words.get_object_by_index(word_index)
        assert active, 'update on active word'
        self._lxc.update_matrix_on_given_row(word_index, scalar)

    def update_word_category_connection(self, word: NewWord, category: NewCategory, update: Callable[[float], float]):
        word_index = self._words.get_object_index(word)
        _, active = self._words.get_object_by_index(word_index)
        assert active, 'update on active word'
        category_index = self._categories.get_object_index(category)
        _, active = self._categories.get_object_by_index(category_index)
        assert active, 'update on active category'
        self._lxc.update_cell(word_index, category_index, update)

    def get_connection(self, word: NewWord, category: NewCategory) -> float:
        word_index = self._words.get_object_index(word)
        category_index = self._categories.get_object_index(category)
        return self._lxc(word_index, category_index)

    def word_meanings(self, stimuli: List[Stimulus], calculator: Calculator) -> Dict[NewWord, List[Stimulus]]:
        # [f] = {q : SUM L(f,c)*<c|R_q> > 0} = {q : L(f,c) > 0 and <c|R_q> > 0}

        # work on active connections only
        self.remove_nonactive_words()
        self.remove_nonactive_categories()

        word2meanings = {}
        non_zero_coordinates = [*zip(*np.nonzero(self._lxc.reduce()))]
        for word_index, category_index in non_zero_coordinates:
            word, _ = self._words.get_object_by_index(word_index)
            if word not in word2meanings.keys():
                word2meanings[word] = []
            category, _ = self._categories.get_object_by_index(category_index)
            word_meaning = word2meanings[word]
            word_meaning += [q for q in stimuli if category.response(q, calculator) > 0]

        word2meanings_distinct_and_sorted = {}
        for word, meaning in word2meanings.items():
            distinct_and_sorted = list(set(meaning))
            word2meanings_distinct_and_sorted[word] = sorted(distinct_and_sorted)

        return word2meanings
