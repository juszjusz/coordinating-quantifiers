import dataclasses
import logging
from copy import copy
from fractions import Fraction
from itertools import groupby
from typing import Callable, List, Dict, Union, Tuple

import numpy as np
from tqdm import tqdm

from calculator import Calculator, StimulusContext, Stimulus
from matrix_datastructure import Matrix, One2OneMapping

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class NewCategory:
    def __init__(self, category_id: int):
        self.category_id = category_id
        self._weights = []
        self._reactive_units = []

    @classmethod
    def init_from_stimulus(cls, stimulus: Stimulus):
        return cls.init_from_stimuli([(.5, stimulus)])

    @classmethod
    def init_from_stimuli(cls, wxr: List[Tuple[float, Stimulus]]):
        new_instance = cls(0)
        new_instance._weights = [w for w, _ in wxr]
        new_instance._reactive_units = [r for _, r in wxr]
        return new_instance

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
        return str(self)

    def __str__(self):
        weights = [round(w, 2) for w in self._weights]
        ru = [r for r in self._reactive_units]
        wXr = str([*zip(weights, ru)])
        return f'id: {self.category_id}; wXr: {wXr}'

    def reactive_units(self):
        return self._reactive_units

    def weights(self):
        return self._weights

    def response(self, stimulus: Stimulus, calculator: Calculator):
        return sum([weight * calculator.dot_product(ru_value, stimulus) for weight, ru_value in
                    zip(self._weights, self._reactive_units)])

    def response_all(self, calculator: Calculator):
        all_stimuli_response = calculator.dot_product_all(self._reactive_units)
        all_stimuli_response = all_stimuli_response * self._weights
        return np.sum(all_stimuli_response, axis=1)

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
        return FUN([weight * calculator.pdf(ru) for weight, ru in zip(self._weights, self._reactive_units)])


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

    def __str__(self):
        return str(self.word_id)

    def __repr__(self):
        return str(self)


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
    def update_call(agent, *args):
        if update.__name__ == 'next_step':
            agent.updates_history.append([])
        current_step = agent.updates_history[-1]
        args_copy = [copy(a) for a in args]
        # logger.info(update.__name__)
        current_step.append([update.__name__, args_copy])
        update(agent, *args)

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
        return f'agent {self.agent_id}'

    @staticmethod
    def snapshot(agent):
        snapshot = NewAgent(agent_id=agent.agent_id, calculator=agent._calculator, game_params=agent._game_params)
        snapshot._lxc = copy(agent._lxc)
        return snapshot

    @staticmethod
    def recreate_from_history(agent_id: int, calculator: Calculator, game_params: GameParams, updates_history: List,
                              snapshot_rate: int = 100):
        snapshots = []
        recreated_agent = NewAgent(agent_id=agent_id, calculator=calculator, game_params=game_params)

        for step, step_updates in enumerate(tqdm(updates_history, f'recreating agent {agent_id} by updates')):
            for method_name, args in step_updates:
                msg = f'meth: {method_name} {args}'
                logger.debug(msg)

                agent_method = getattr(recreated_agent, method_name)
                agent_method(*args)

            if step % snapshot_rate == 0:
                snapshots.append((step, NewAgent.snapshot(recreated_agent)))

        if (len(updates_history) - 1) % snapshot_rate != 0:
            # recreate agent's last state
            snapshots.append(((len(updates_history) - 1), NewAgent.snapshot(recreated_agent)))

        return snapshots

    @staticmethod
    def to_dict(agent) -> Dict:
        agent._lxc.remove_nonactive_categories()
        agent._lxc.remove_non_responsive_words()

        words = [{'word_id': w.word_id, 'originated_from_category': {
            'reactive_units': [[r.numerator, r.denominator] if isinstance(r, Fraction) else r for r in
                               w.originated_from_category.reactive_units()],
            'weights': [round(w, 3) for w in w.originated_from_category.weights()]
        }}
                 for w in agent.get_words()]

        categories = [{'category_id': category.category_id,
                       'reactive_units': [[r.numerator, r.denominator] if isinstance(r, Fraction) else r for r in
                                          category.reactive_units()],
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
        return len(self._lxc.get_responsive_categories()) > 0

    def get_words(self) -> List[NewWord]:
        return self._lxc.get_responsive_words()

    # def get_active_words(self, stimuli: List[Stimulus]) -> List[NewWord]:
    #     response_category_maximizers = [self.get_most_responsive_category(s) for s in stimuli]
    #     response_category_maximizers = [c for c in response_category_maximizers]
    #     active_lexicon = [self.get_most_connected_word(c) for c in response_category_maximizers]
    #     active_lexicon = [w for w in active_lexicon if w is not None]
    #     return list(set(active_lexicon))

    def compute_active_words(self) -> List[NewWord]:
        response_category_maximizers = self.get_most_responsive_category_over_all_stimuli()
        active_lexicon = [self.get_most_connected_word(c) for c in response_category_maximizers]
        active_lexicon = [w for w in active_lexicon if w is not None]
        return list(set(active_lexicon))

    def get_categories(self) -> List[NewCategory]:
        return self._lxc.get_responsive_categories()

    def get_most_connected_word(self, category: NewCategory, activation_threshold=0) -> Union[NewWord, None]:
        return self._lxc.get_most_connected_word(category, activation_threshold)

    def get_most_connected_category(self, word: NewWord, activation_threshold=0) -> Union[NewCategory, None]:
        return self._lxc.get_most_connected_category(word, activation_threshold)

    def get_most_responsive_category(self, stimulus: Stimulus) -> Union[NewCategory, None]:
        active_categories = self._lxc.get_responsive_categories()
        responses = [c.response(stimulus, self._calculator) for c in active_categories]
        response_argmax = np.argmax(responses)
        return active_categories[response_argmax]

    def get_most_responsive_category_over_all_stimuli(self) -> List[NewCategory]:
        active_categories = self._lxc.get_responsive_categories()
        if len(active_categories) > 0:
            responses = [c.response_all(self._calculator) for c in active_categories]
            response_maximizers = np.argmax(responses, axis=0)
            return list(set(active_categories[maximizer] for maximizer in response_maximizers))
        else:
            return []

    def knows_word(self, w: NewWord):
        active_words = self._lxc.get_responsive_words()
        return w in active_words

    @register_agent_update_operation
    def forget_categories(self, category_in_use: NewCategory):
        self._lxc.forget_categories(category_in_use, self._game_params.alpha, self._game_params.super_alpha)

    @register_agent_update_operation
    def forget_words(self, super_alpha=.01):
        # todo use super_alpha from game parrams ?
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
        self._lxc.update_word_category_connection(word, category, lambda v: v + self._game_params.delta_inc * v)
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
            category = self.get_most_responsive_category(stimulus)
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

    def compute_word_meanings(self) -> Dict[NewWord, List[bool]]:
        active_words = self.compute_active_words()
        return self._lxc.compute_word_meanings(active_words, self._calculator)

    def compute_word_pragmatic_meanings(self, stimuli: List[Stimulus]) -> Dict[NewWord, List[bool]]:
        return self._lxc.compute_word_pragmatic_meanings(stimuli, self._calculator)

    @staticmethod
    def is_monotone_new(stimuli_activations: List[bool]):
        return NewAgent._compute_number_of_inflections(stimuli_activations) == 1

    @staticmethod
    def is_convex_new(stimuli_activations: List[bool]):
        return NewAgent._compute_number_of_inflections(stimuli_activations) <= 2

    @staticmethod
    def _compute_number_of_inflections(activations: List[bool]):
        return len([current for current, next in zip(activations, activations[1:]) if current != next])

    def get_discriminative_success(self):
        return self._discriminative_success_means

    def get_communicative_success1(self):
        return self._communicative_success1

    def get_communicative_success2(self):
        return self._communicative_success2


class SimpleCounter:
    def __init__(self) -> None:
        self._counter = 0

    def __repr__(self):
        return 'offset: ' + str(self._counter)

    def __call__(self) -> int:
        i = self._counter
        self._counter += 1
        return i


class LxC:
    def __init__(self, row: int, col: int):
        self._categories = One2OneMapping()
        self._words = One2OneMapping()
        self._lxc = Matrix(row, col)

    def __copy__(self):
        self.remove_nonactive_categories()
        self.remove_non_responsive_words()

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
        return self._categories.get_managed_object(c)

    def get_matrix(self) -> np.ndarray:
        return self._lxc.reduce()

    def add_new_or_reactivate_word(self, w: NewWord):
        if w in self._words.nonactive_elements():
            # activate previously deactivated word
            word_index = self._words.get_object_index(w)
            self._words.reactivate_element_at_index(word_index)
        else:
            # add new word
            self.remove_non_responsive_words()
            self._words.add_new_element(w)
            self._lxc.add_new_row()

    def add_new_category(self, new_category: NewCategory):
        self._categories.add_new_element(new_category)
        self._lxc.add_new_col()

    def get_most_connected_word(self, category: NewCategory, activation_threshold=0) -> Union[NewWord, None]:
        category_index = self._categories.get_object_index(category)
        word_category_maximizer_index = self._lxc.get_col_argmax(category_index)
        wXc_value = self._lxc[word_category_maximizer_index, category_index]
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
        wXc_value = self._lxc[word_index, category_word_maximizer_index]
        if wXc_value > activation_threshold:
            category_word_maximizer, active = self._categories.get_object_by_index(category_word_maximizer_index)
            assert active, 'WxC > activation_th -> category active'
            return category_word_maximizer
        else:
            return None

    def get_responsive_categories(self) -> List[NewCategory]:
        return self._categories.active_elements()

    def get_responsive_words(self) -> List[NewWord]:
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
            self.remove_non_responsive_words()

    def remove_non_responsive_words(self):
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
        return self._lxc[word_index, category_index]

    def compute_word_meanings(self, active_words: List[NewWord], calculator: Calculator) -> Dict[NewWord, List[bool]]:
        # [f] = {q : SUM L(f,c)*<c|R_q> > 0} = {q : L(f,c) > 0 and <c|R_q> > 0}
        self.remove_non_responsive_words()
        self.remove_nonactive_categories()

        activate_words_indices = set(self._words.get_object_index(w) for w in active_words)
        non_zero_wXc_connections = [*zip(*np.nonzero(self._lxc.reduce()))]
        non_zero_wXc_connections = [(w_index, c_index) for w_index, c_index in non_zero_wXc_connections if
                                    w_index in activate_words_indices]
        non_zero_wXc_connections = [(word_index, [*v]) for word_index, v in
                                    groupby(non_zero_wXc_connections, key=lambda wXc: wXc[0])]

        word2meanings = {}
        for word_index, categories in non_zero_wXc_connections:
            word, _ = self._words.get_object_by_index(word_index)
            categories = [self._categories.get_object_by_index(category_index) for _, category_index in categories]
            word2meanings[word] = np.sum([category.response_all(calculator) for category, _ in categories], axis=0)

        return {word: calculator.activation_from_responses(responses) for word, responses in word2meanings.items()}

    def compute_word_pragmatic_meanings(self, stimuli: List[Stimulus], calculator: Calculator) -> Dict[
        NewWord, List[bool]]:
        # work on active connections only
        self.remove_non_responsive_words()
        self.remove_nonactive_categories()

        categories = np.array(self._categories.active_elements())

        responses = [category.response_all(calculator) for category in categories]
        stimuli_response_maximizers = np.argmax(responses, axis=0)

        category2stimuli = [(category, stimuli) for stimuli, category in enumerate(stimuli_response_maximizers)]

        category2stimuli = sorted(category2stimuli, key=lambda c2s: c2s[0])
        category2stimuli = {category: [s for c, s in v] for category, v in
                            groupby(category2stimuli, key=lambda c2s: c2s[0])}

        stimuli_response_maximizers = set(stimuli_response_maximizers)
        word2category = list(set((self.get_most_connected_word(categories[i]), i) for i in stimuli_response_maximizers))
        word2category = [(w, category_index) for w, category_index in word2category if w is not None]
        word2meaning = {}
        for word, category_index in word2category:
            active_stimuli = category2stimuli[category_index]
            activations = np.array([False] * len(stimuli))
            activations[active_stimuli] = True
            word2meaning = {word: activations}
        return word2meaning
