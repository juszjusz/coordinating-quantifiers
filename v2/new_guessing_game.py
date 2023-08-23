from collections import deque
from fractions import Fraction
from threading import Lock
from scipy.sparse import lil_matrix
import graphviz
import argparse
import dataclasses
import logging
from typing import Dict, Tuple, Union, List, Callable, Any

from numpy import ndarray
from numpy.random import RandomState

import numpy as np

from v2.calculator import NumericCalculator, QuotientCalculator, Calculator, NewAbstractStimulus
from v2.category import NewCategory

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def flip_a_coin_random_function(seed=0) -> Callable[[], int]:
    r = np.random.RandomState(seed=seed)

    def flip_a_coin() -> int:
        return r.binomial(1, .5)

    return flip_a_coin


def shuffle_list_random_function(seed=0) -> Callable[[List], None]:
    r = np.random.RandomState(seed=seed)

    def shuffle_list(l: List) -> None:
        r.shuffle(l)

    return shuffle_list


def pick_element_random_function(seed=0) -> Callable[[List], Any]:
    r = np.random.RandomState(seed=seed)

    def pick_random_value(l: List) -> Any:
        i = r.randint(len(l))
        return l[i]

    return pick_random_value


@dataclasses.dataclass
class GameParams:
    population_size: int
    steps: int
    stimulus: str
    max_num: int
    runs: int
    guessing_game_2: bool
    in_mem_calculus_path: str
    seed: int
    discriminative_threshold: float
    discriminative_history_length: int
    delta_inc: float  # params['delta_inc']
    delta_dec: float  # params['delta_dec']
    delta_inh: float  # params['delta_inh']
    alpha: float  # params['alpha']  # forgetting
    beta: float  # params['beta']  # learning rate
    super_alpha: float  # params['super_alpha']


@dataclasses.dataclass
class NewWord:
    word_id: int
    active = True

    def __hash__(self):
        return self.word_id

    def deactivate(self):
        self.active = False


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


class ThreadSafeWordFactory:

    def __init__(self) -> None:
        self._counter = 0
        self._lock = Lock()

    def __call__(self) -> NewWord:
        with self._lock:
            w = NewWord(self._counter)
            self._counter += 1
            return w


class ConnectionMatrixLxC:
    def __init__(self, max_row: int, max_col: int):
        self._square_matrix = np.zeros((max_row, max_col))
        self._row = 0
        self._col = 0

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
    def __init__(self, agent_id: int, lxc_max_size: int, game_params: GameParams):
        self.agent_id = agent_id
        self._game_params = game_params
        row = int(lxc_max_size / 5)
        col = int(np.sqrt(lxc_max_size))
        self._lxc = ConnectionMatrixLxC(row, col)
        self._lexicon: List[NewWord] = []
        self._categories: List[NewCategory] = []
        self._lex2index = {}
        self._discriminative_success = deque(maxlen=self._game_params.discriminative_history_length)
        self._discriminative_success_mean = 0.

    def __repr__(self):
        actual_lexicon = len(self._lexicon)
        actual_categories = len(self._categories)
        return str(self._lxc.reduce(actual_lexicon, actual_categories))

    @staticmethod
    def to_dict(agent) -> Dict:
        words = [{'word_id': w.word_id, 'word_position': agent._lex2index[w], 'active': w.active}
                 for w in agent._lexicon]

        categories = [{'category_id': category.category_id, 'is_active': category.is_active,
                       'reactive_units': category.reactive_units(),
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

    def add_discrimination_success(self):
        self._discriminative_success.append(True)
        self._discriminative_success_mean = np.mean(self._discriminative_success)

    def add_discrimination_failure(self):
        self._discriminative_success.append(False)
        self._discriminative_success_mean = np.mean(self._discriminative_success)

    # based on how much the word meaning covers the category
    def csimilarity(self, word: NewWord, category: NewCategory, calculator: Calculator):
        area = category.union(calculator)
        # omit multiplication by x_delta because all we need is ratio: coverage/area:
        word_meaning = self.word_meaning(word, calculator)
        coverage = np.minimum(word_meaning, area)

        return sum(coverage) / sum(area)

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


class NewPopulation:

    def __init__(self, population_size: int, steps: int, game_params: GameParams, shuffle_list: Callable[[List], Any]):
        assert population_size % 2 == 0, 'each agent must be paired'
        self._shuffle_list = shuffle_list
        self._agents = [NewAgent(agent_id, steps, game_params) for agent_id in range(population_size)]

    def select_pairs(self) -> List[Tuple[NewAgent, NewAgent]]:
        self._shuffle_list(self._agents)
        return [(self._agents[i], self._agents[i + 1]) for i in range(0, len(self._agents), 2)]

    def __iter__(self):
        return iter(self._agents)

    def __len__(self):
        return len(self._agents)


class GuessingGameAction:
    def __call__(self, agent: NewAgent, context: Tuple[NewAbstractStimulus, NewAbstractStimulus], data_envelope: Dict,
                 **kwargs) -> str:
        pass


def select_speaker(speaker: NewAgent, _: NewAgent) -> NewAgent:
    return speaker


def select_hearer(_: NewAgent, hearer: NewAgent) -> NewAgent:
    return hearer


class DiscriminationGameAction(GuessingGameAction):

    def __init__(self, on_no_category: str,
                 on_no_noticeable_difference: str,
                 on_no_discrimination: str,
                 on_success: str,
                 selected_category_path: str,
                 calculator: Calculator,
                 stats: AggregatedGameResultStats):
        self.on_no_category = on_no_category
        self.on_no_noticeable_difference = on_no_noticeable_difference
        self.on_no_discrimination = on_no_discrimination
        self.on_success = on_success

        self.selected_category_path = selected_category_path

        self.calculator = calculator
        self.stats = stats

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, topic: int) -> str:
        if not agent.has_categories():
            logging.debug('no category {}({})'.format(agent, agent.agent_id))
            agent.learn_stimulus(context[topic], self.calculator)
            agent.add_discrimination_failure()
            self.stats.add_discrimination_failure(agent)
            return self.on_no_category

        s1, s2 = context

        if not s1.is_noticeably_different_from(s2):
            return self.on_no_noticeable_difference

        category1 = agent.get_best_matching_category(s1, self.calculator)
        category2 = agent.get_best_matching_category(s2, self.calculator)

        if category1 == category2:
            logging.debug('no category {}({})'.format(agent, agent.agent_id))
            agent.learn_stimulus(context[topic], self.calculator)
            agent.add_discrimination_failure()
            self.stats.add_discrimination_failure(agent)
            return self.on_no_discrimination

        winning_category = [category1, category2][topic]

        # todo można to wbić do jednej metody, dwie metody występują tylko tutaj
        agent.reinforce_category(winning_category, context[topic], calculator=self.calculator)
        agent.forget_categories(winning_category)

        data_envelope[self.selected_category_path] = winning_category

        agent.add_discrimination_success()
        self.stats.add_discrimination_success(agent)
        return self.on_success


class PickMostConnectedWord(GuessingGameAction):
    """ 3. The speaker searches for words f in DS which are associated with cS , i.e.
    such that LS (f, cS ) > 0.
    If no associated words are found (i.e., LS (f, cS ) = 0 for all f ∈ DS ) or
    LS is empty, the speaker creates a new word f (i.e., DS := DS ∪ {f }), sets
    LS (f, cS ) = 0.5 and utters f .
    Now suppose that some associated words are found. Let f1,..., fn be all
    words associated with cS . The speaker chooses f from f1,..., fn such that
    LS (f, cS ) ≥ LS (fi, cS ), for i = 1, 2,..., n, and conveys f to the hearer. The
    choice of such a word is consistent with the concept of pragmatic meaning,
    defined later in this section."""

    def __init__(self, on_success: str,
                 selected_word_path: str,
                 new_word: Callable[[], NewWord]):
        self.select_word_for_category = on_success
        self.selected_word_path = selected_word_path
        self.new_word = new_word

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, category: NewCategory) -> str:
        word = agent.get_most_connected_word(category)

        if word is None:
            logger.debug("%s(%d) introduces new word \"%s\"" % (agent, agent.agent_id, word))
            logger.debug("%s(%d) associates \"%s\" with his category" % (agent, agent.agent_id, word))
            word = self.new_word()
            agent.add_new_word(word)
            agent.learn_word_category(word, category)

        logger.debug("Agent(%d) says: %s" % (agent.agent_id, word))
        data_envelope[self.selected_word_path] = word
        return self.select_word_for_category


class PickMostConnectedCategoryAction(GuessingGameAction):
    """ 4. The hearer looks up f in her lexicon DH.
    If f /∈ DH, the game fails and the topic is revealed to the hearer. The
    repair mechanism is as follows. First, the hearer adds f to her lexicon (i.e.,
    DH := DH ∪ {f}). Next, the hearer plays the discrimination game to see
    whether or not she has a category capable of discriminating the topic. If
    one is found, say c, the hearer creates an association between f and c with
    the initial strength of 0.5 (i.e., LH (f, c) = 0.5).
    Suppose the hearer finds f in DH . Let c1, c2, ... , ck be the list of all
    categories associated with f in LH (i.e., LH (f, ci) > 0 for i = 1, 2, ... , k).
    The hearer chooses cH from c1, ... , ck such that L(f, cH ) ≥ L(f, ci) for
    i = 1, 2, ... , k. The hearer points to the stimulus, denoted by qH , that
    generates the highest response for cH (i.e., qH = argmax q∈{q1,q2}〈cH |Rq〉)."""

    def __init__(self, on_unknown_word_or_no_associated_category: str, on_known_word: str, selected_category_path: str):
        self.on_unknown_word_or_no_associated_category = on_unknown_word_or_no_associated_category
        self.on_known_word = on_known_word
        self.selected_category_path = selected_category_path

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, word: NewWord):
        if not agent.knows_word(word):
            agent.add_new_word(word)
            return self.on_unknown_word_or_no_associated_category

        category = agent.get_most_connected_category(word)

        if category is None:
            return self.on_unknown_word_or_no_associated_category

        data_envelope[self.selected_category_path] = category

        return self.on_known_word


class SelectAndCompareTopic(GuessingGameAction):
    """ 5. The topic is revealed to the hearer. If qS = qH , i.e., the topic is the same as
        the guess of the hearer, the game is successful. Otherwise, the game fails """

    def __init__(self, on_success: str, on_failure: str, flip_a_coin: Callable[[], int],
                 calculator: Calculator, stats: AggregatedGameResultStats):
        self.on_success = on_success
        self.on_failure = on_failure

        self.flip_a_coin = flip_a_coin
        self._calculator = calculator
        self.stats = stats

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, category: NewCategory, topic: int) -> str:
        selected = category.select(context, self._calculator)

        if selected is None:
            selected = self.flip_a_coin()

        if selected == topic:
            self.stats.add_communication1_success(agent)
            return self.on_success
        else:
            self.stats.add_communication1_failure(agent)
            return self.on_failure


class CompareWordsAction(GuessingGameAction):
    def __init__(self, on_equal_words: str, on_different_words: str, stats: AggregatedGameResultStats):
        self.on_equal_words = on_equal_words
        self.on_different_words = on_different_words
        self.stats = stats

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, speaker_word: NewWord,
                 hearer_word: NewWord) -> str:
        if speaker_word == hearer_word:
            self.stats.add_communication2_success(agent)
            return self.on_equal_words
        else:
            self.stats.add_communication2_failure(agent)
            return self.on_different_words


class SuccessAction(GuessingGameAction):
    def __init__(self, on_success: str):
        self.on_success = on_success

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, word: NewWord, category: NewCategory) -> str:
        agent.update_on_success(word, category)
        return self.on_success


class FailureAction(GuessingGameAction):
    def __init__(self, on_success: str):
        self.on_success = on_success

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, word: NewWord, category: NewCategory) -> str:
        agent.update_on_failure(word, category)
        return self.on_success


class LearnWordCategoryAction(GuessingGameAction):
    def __init__(self, on_success: str):
        self.on_success = on_success

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, word: NewWord, category: NewCategory) -> str:
        agent.learn_word_category(word, category)
        return self.on_success


class CompleteAction:
    def __init__(self, on_success: str):
        self.on_success = on_success

    def __call__(self, agent: NewAgent, context, data_envelope: Dict) -> str:
        agent.forget_words()
        return self.on_success


def game_graph_with_stage_7(calculator: Calculator, game_params: GameParams, flip_a_coin: Callable[[], int]):
    stats = AggregatedGameResultStats(game_params=game_params)
    new_word = ThreadSafeWordFactory()

    return {'2_SPEAKER_DISCRIMINATION_GAME':
        {'action': DiscriminationGameAction(
            on_no_category='SPEAKER_COMPLETE',
            on_no_noticeable_difference='SPEAKER_COMPLETE',
            on_no_discrimination='SPEAKER_COMPLETE',
            on_success='3_SPEAKER_PICKUPS_WORD_FOR_CATEGORY',
            selected_category_path='SPEAKER.category',
            calculator=calculator,
            stats=stats
        ), 'agent': 'SPEAKER', 'args': ['topic']},

        '3_SPEAKER_PICKUPS_WORD_FOR_CATEGORY':
            {'action': PickMostConnectedWord(
                on_success='4_SPEAKER_CONVEYS_WORD_TO_THE_HEARER',
                selected_word_path='SPEAKER.word',
                new_word=new_word
            ), 'agent': 'SPEAKER', 'args': ['SPEAKER.category']},

        '4_SPEAKER_CONVEYS_WORD_TO_THE_HEARER':
            {'action': PickMostConnectedCategoryAction(
                on_unknown_word_or_no_associated_category='4_1_SPEAKER_CONVEYS_WORD_TO_THE_HEARER_ON_UNKNOWN_WORD',
                on_known_word='5_CHECK_TOPIC',
                selected_category_path='HEARER.category'
            ), 'agent': 'HEARER', 'args': ['SPEAKER.word']},

        '4_1_SPEAKER_CONVEYS_WORD_TO_THE_HEARER_ON_UNKNOWN_WORD':
            {'action': DiscriminationGameAction(
                on_no_category='SPEAKER_COMPLETE',
                on_no_noticeable_difference='SPEAKER_COMPLETE',
                on_no_discrimination='SPEAKER_COMPLETE',
                on_success='4_2_HEARER_LEARNS_WORD_CATEGORY',
                selected_category_path='HEARER.category',
                calculator=calculator,
                stats=stats
            ), 'agent': 'HEARER', 'args': ['topic']},

        '4_2_HEARER_LEARNS_WORD_CATEGORY':
            {'action': LearnWordCategoryAction(
                on_success='SPEAKER_COMPLETE'
            ), 'agent': 'HEARER', 'args': ['SPEAKER.word', 'HEARER.category']},

        '6_HEARER_PICKUPS_WORD_FOR_CATEGORY':
            {'action': PickMostConnectedWord(
                on_success='7_HEARER_COMPARES_WORD',
                selected_word_path='HEARER.word',
                new_word=new_word
            ), 'agent': 'HEARER', 'args': ['HEARER.category']},

        '7_HEARER_COMPARES_WORD':
            {'action': CompareWordsAction(on_equal_words='SPEAKER_SUCCESS', on_different_words='SPEAKER_FAILURE',
                                          stats=stats),
             'agent': 'HEARER', 'args': ['SPEAKER.word', 'HEARER.word']},

        '5_CHECK_TOPIC':
            {'action': SelectAndCompareTopic(on_success='SPEAKER_SUCCESS', on_failure='SPEAKER_FAILURE', stats=stats,
                                             flip_a_coin=flip_a_coin, calculator=calculator),
             'agent': 'HEARER', 'args': ['HEARER.category', 'topic']},

        'SPEAKER_SUCCESS':
            {'action': SuccessAction(on_success='HEARER_SUCCESS'),
             'agent': 'SPEAKER', 'args': ['SPEAKER.word', 'SPEAKER.category']},

        'HEARER_SUCCESS':
            {'action': SuccessAction(on_success='SPEAKER_COMPLETE'),
             'agent': 'HEARER', 'args': ['SPEAKER.word', 'HEARER.category']},

        'SPEAKER_FAILURE':
            {'action': FailureAction(on_success='HEARER_FAILURE'),
             'agent': 'SPEAKER', 'args': ['SPEAKER.word', 'SPEAKER.category']},

        'HEARER_FAILURE':
            {'action': FailureAction(on_success='SPEAKER_COMPLETE'),
             'agent': 'HEARER', 'args': ['SPEAKER.word', 'HEARER.category']},

        'SPEAKER_SUCCESS_7':
            {'action': SuccessAction(on_success='HEARER_SUCCESS'),
             'agent': 'SPEAKER', 'args': []},

        'HEARER_SUCCESS_7':
            {'action': SuccessAction(on_success='SPEAKER_COMPLETE'),
             'agent': 'HEARER', 'args': []},

        'SPEAKER_COMPLETE': {'action': CompleteAction(on_success='HEARER_COMPLETE'), 'agent': 'SPEAKER', 'args': []},

        'HEARER_COMPLETE': {'action': CompleteAction(on_success='NEXT_STEP'), 'agent': 'HEARER', 'args': []},

        'NEXT_STEP': {'action': None, 'agent': None, 'args': []}
    }


def run_simulation(steps: int, population_size: int,
                   shuffle_list: Callable[[List], None],
                   game_params: GameParams,
                   context_constructor: Callable[[], Tuple[NewAbstractStimulus, NewAbstractStimulus]],
                   game_graph):
    population = NewPopulation(population_size, steps, game_params, shuffle_list)

    for step in range(steps):
        paired_agents = population.select_pairs()

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

                state_name = action(agent, context, data_envelope, *args)
                state = game_graph[state_name]
                action = state['action']
                agent_name = state['agent']
                arg_names = state['args']
                args = [data_envelope[a] for a in arg_names]

                logger.debug(data_envelope)

    return population


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='quantifiers simulation')

    # parser.add_argument('--simulation_name', '-sn', help='simulation name', type=str, default='test')
    parser.add_argument('--population_size', '-p', help='population size', type=int, default=10)
    parser.add_argument('--stimulus', '-stm', help='quotient or numeric', type=str, default='numeric',
                        choices=['quotient', 'numeric'])
    parser.add_argument('--max_num', '-mn', help='max number for numerics or max denominator for quotients', type=int,
                        default=100)
    parser.add_argument('--discriminative_threshold', '-dt', help='discriminative threshold', type=float, default=.95)
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
    parser.add_argument('--steps', '-s', help='number of steps', type=int, default=100)
    parser.add_argument('--runs', '-r', help='number of runs', type=int, default=1)
    parser.add_argument('--guessing_game_2', '-gg2', help='is the second stage of the guessing game on',
                        action='store_true')
    parser.add_argument('--in_mem_calculus_path', '-path', help='path to precomputed integrals', type=str,
                        default='../inmemory_calculus')
    parser.add_argument('--seed', help='set seed value to replicate a random values', type=int, default=1)

    parsed_params = vars(parser.parse_args())

    log_levels = {'debug': logging.DEBUG, 'info': logging.INFO}
    # logging.basicConfig(stream=sys.stderr, level=log_levels[parsed_params['log_level']])
    # load_inmemory_calculus(parsed_params['in_mem_calculus_path'], parsed_params['stimulus'])
    calculator = {'numeric': NumericCalculator.load_from_file(),
                  'quotient': QuotientCalculator.load_from_file()}[parsed_params['stimulus']]

    seed = 0  # parsed_params['seed']

    shuffle_list = shuffle_list_random_function(seed=seed)
    flip_a_coin = flip_a_coin_random_function(seed=seed)
    pick_element = pick_element_random_function(seed=seed)

    context_constructor = calculator.context_factory(pick_element=pick_element)

    population_size = 20  # parsed_params['population_size']
    steps = 3000  # parsed_params['steps']

    game_params = GameParams(**parsed_params)

    # population = NewPopulation(population_size, steps, shuffle_list)
    # game_graph = graphviz.Digraph()

    population = run_simulation(steps, population_size, shuffle_list, game_params,
                                context_constructor, game_graph_with_stage_7(calculator, game_params, flip_a_coin))
    print([len(NewAgent.to_dict(a)['categories']) for a in population])
    print([len(NewAgent.to_dict(a)['words']) for a in population])
    # categories cnt after 1000x
    # [16, 13, 13, 21, 12, 29, 17, 14, 23, 11, 12, 34, 19, 24, 12, 28, 9, 31, 24, 24]
    # words cnt after 1000x
    # [91, 92, 98, 94, 89, 102, 95, 94, 105, 95, 97, 98, 92, 93, 95, 101, 96, 102, 102, 88]

    # game_graph.node('SPEAKER_DISCRIMINATION_GAME', label='root node', attrs='speaker')
    # game_graph.node('SPEAKER_NO_CATEGORY_AFTER_DISCRIMINATION_GAME', attrs='speaker')
    # game_graph.node('SPEAKER_NO_DIFFERENCE_AFTER_DISCRIMINATION_GAME')
    # game_graph.node('SPEAKER_DISCRIMINATION_GAME_SUCCESS')
    #
    # game_graph.edge('SPEAKER_DISCRIMINATION_GAME', 'SPEAKER_NO_CATEGORY_AFTER_DISCRIMINATION_GAME')
    # game_graph.edge('SPEAKER_DISCRIMINATION_GAME', 'SPEAKER_NO_DIFFERENCE_AFTER_DISCRIMINATION_GAME')
    # game_graph.edge('SPEAKER_DISCRIMINATION_GAME', 'SPEAKER_DISCRIMINATION_GAME_SUCCESS')
    #
    # print(game_graph.source)
