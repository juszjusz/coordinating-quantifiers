from __future__ import division  # force python 3 division in python 2
import graphviz
import argparse
import dataclasses
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Union, List, Callable

from numpy.random import RandomState

import numpy as np

from inmemory_calculus import inmem, load_inmemory_calculus
from stimulus import QuotientBasedStimulusFactory, NumericBasedStimulusFactory, ContextFactory, AbstractStimulus
from v2.category import NewCategory

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GameParams:
    population_size: int
    steps: int
    stimulus: AbstractStimulus
    max_num: int
    runs: int
    guessing_game_2: bool
    in_mem_calculus_path: str
    seed: int
    discriminative_threshold: float
    delta_inc: float  # params['delta_inc']
    delta_dec: float  # params['delta_dec']
    delta_inh: float  # params['delta_inh']
    discriminative_threshold: float  # params['discriminative_threshold']
    alpha: float  # params['alpha']  # forgetting
    beta: float  # params['beta']  # learning rate
    super_alpha: float  # params['super_alpha']


@dataclasses.dataclass
class NewWord:
    word_id: int

    def __hash__(self):
        return self.word_id


class NewWordFactory:

    def __init__(self, r: RandomState) -> None:
        super().__init__()
        self.random = r

    def __call__(self) -> NewWord:
        word_new_id = self.random.randint(999_999_999_999)
        return NewWord(word_new_id)


new_word = NewWordFactory(RandomState(0))


class ConnectionMatrixLxC:
    def __init__(self, size: int):
        self.square_matrix = np.zeros((size, size))

    def to_ndarray(self):
        return self.square_matrix

    def __call__(self, row, col) -> float:
        return self.square_matrix[row, col]

    def rows(self):
        return self.square_matrix.shape[0]

    def cols(self):
        return self.square_matrix.shape[1]

    def get_row_argmax(self, row_index) -> int:
        return np.argmax(self.square_matrix, axis=0)[row_index]

    def get_col_argmax(self, col_index) -> int:
        return np.argmax(self.square_matrix, axis=1)[col_index]

    def update_cell(self, row: int, column: int, update: Callable[[float], float]):
        recomputed_value = update(self.square_matrix[row, column])
        self.square_matrix[row, column] = recomputed_value

    def update_matrix_on_indices(self, row_indices: List[int], column_indices: List[int], scalar: float):
        updated_cells = self.square_matrix[row_indices, column_indices]
        self.square_matrix[row_indices, column_indices] += scalar * updated_cells


class NewAgent:
    def __init__(self, agent_id: int, lxc_max_size: int):
        self.agent_id = agent_id

        self.__lxc = ConnectionMatrixLxC(lxc_max_size)

        self.__lexicon: List[NewWord] = []
        self.__categories: List[NewCategory] = []
        self.__cat2index = {}
        self.__lex2index = {}

    def __repr__(self):
        return str(self.__lxc)

    def get_most_connected_word(self, category: NewCategory, activation_threshold=0) -> Union[NewWord, None]:
        category_index = self.__cat2index[category]
        category_argmax = self.__lxc.get_col_argmax(category_index)
        value = self.__lxc(category_argmax, category_index)
        if value >= activation_threshold:
            return self.__lexicon[category_argmax]
        else:
            return None

    def get_most_connected_category(self, word: NewWord, activation_threshold=0) -> Union[NewCategory, None]:
        word_index = self.__lex2index[word]
        word_argmax = self.__lxc.get_row_argmax(word_index)
        value = self.__lxc(word_index, word_argmax)
        if value >= activation_threshold:
            return self.__categories[word_argmax]
        else:
            return None

    def has_categories(self) -> bool:
        return len(self.__categories) > 0

    def get_best_matching_category(self, stimulus: AbstractStimulus) -> NewCategory:
        responses = [c.response(stimulus) for c in self.__categories if c.is_active]
        response_argmax = np.argmax(responses)
        return self.__categories[response_argmax]

    def knows_word(self, w: NewWord):
        return w in self.__lex2index.keys()

    def forget_categories(self, category_in_use: NewCategory):
        # category_index = self.get_active_cats().index(category_in_use)
        active_categories = [c for c in self.__categories if c.is_active]

        for c in active_categories:
            c.decrement_weights(game_params.alpha)

        to_forget = [i for i, activate_category in enumerate(active_categories) if
                     activate_category.max_weigth() < game_params.super_alpha and category_in_use != i]

        self.__lxc.update_matrix_on_indices([], to_forget, -1)
        [c.deactivate() for c in to_forget]

    def add_new_word(self, w: NewWord):
        self.__lex2index[w] = len(self.__lexicon)
        self.__lexicon.append(w)

        assert (len(self.__lexicon) <= self.__lxc.rows()), \
            'lexicon size must be at most the size of the lxc matrix height'

    def inhibit_word2categories_connections(self, word: NewWord, except_category: NewCategory):
        word_index = self.__lex2index[word]
        except_category_index = self.__cat2index[except_category]
        indices = [i for i in self.__cat2index.values() if i != except_category_index]
        self.__lxc.update_matrix_on_indices([word_index], indices, -game_params.delta_inh)

    def learn_word_category(self, word: NewWord, category: NewCategory, connection=.5):
        self.__update_connection(word, category, lambda v: connection)

    def update_on_success(self, word: NewWord, category: NewCategory):
        self.__update_connection(word, category, lambda v: v + game_params.delta_inc * v)

    def learn_stimulus(self, stimulus):
        discriminative_success = 0  # todo
        if discriminative_success >= game_params.discriminative_threshold:
            logging.debug("updating category by adding reactive unit centered on %s" % stimulus)
            category = self.get_best_matching_category(stimulus)
            logging.debug("updating category")
            category.add_reactive_unit(stimulus)
        else:
            logging.debug("adding new category centered on %s" % stimulus)
            self.add_new_category(stimulus)

    def add_new_category(self, stimulus, weight=0.5):
        category_id = len(self.__categories)
        new_category = NewCategory(category_id=category_id, seed=(2_147_483_647))
        new_category.add_reactive_unit(stimulus, weight)
        self.__categories.append(new_category)
        assert (len(self.__categories) <= self.__lxc.cols()), \
            'categories size must be at most the size of the lxc matrix height'

    def __update_connection(self, word: NewWord, category: NewCategory, update: Callable[[float], float]):
        word_index = self.__lex2index[word]
        category_index = self.__cat2index[category]
        self.__lxc.update_cell(word_index, category_index, update)


class NewPopulation:

    def __init__(self, population_size: int, steps: int, seed: int):
        assert population_size % 2 == 0, 'each agent must be paired'
        self.__random = RandomState(seed)
        self.__agents = [NewAgent(agent_id, steps) for agent_id in range(population_size)]

    def select_pairs(self) -> List[Tuple[NewAgent, NewAgent]]:
        np.random.shuffle(self.__agents)
        return [(self.__agents[i], self.__agents[i + 1]) for i in range(0, len(self.__agents), 2)]


class GuessingGameAction:
    def __call__(self, agent: NewAgent, context: Tuple[AbstractStimulus, AbstractStimulus],
                 topic: int, data_envelope: Dict) -> str:
        pass

    def is_complete(self):
        return False


def select_speaker(speaker: NewAgent, _: NewAgent) -> NewAgent:
    return speaker


def select_hearer(_: NewAgent, hearer: NewAgent) -> NewAgent:
    return hearer


class DiscriminationGameAction(GuessingGameAction):

    def __init__(self,
                 on_no_category: str,
                 on_no_noticeable_difference: str,
                 on_no_discrimination: str,
                 on_success: str):
        self.on_no_category = on_no_category
        self.on_no_noticeable_difference = on_no_noticeable_difference
        self.on_no_discrimination = on_no_discrimination
        self.on_success = on_success

    def __call__(self, agent, context, topic, data_envelope) -> str:
        # agent: NewAgent = self.select_agent(speaker, hearer)
        # agent.store_ds_result(False) #TODO

        if not agent.has_categories():
            # raise NO_CATEGORY
            return self.on_no_category

        s1, s2 = context

        if not s1.is_noticeably_different_from(s2):
            # raise NO_NOTICEABLE_DIFFERENCE
            return self.on_no_noticeable_difference

        category1 = agent.get_best_matching_category(s1)
        category2 = agent.get_best_matching_category(s2)

        if category1 == category2:
            # raise NO_DISCRIMINATION
            return self.on_no_discrimination

        winning_category = category1 if topic == 0 else category2

        winning_category.reinforce(context[topic], game_params.beta)
        agent.forget_categories(winning_category)
        # agent.switch_ds_result()

        # winning_category_index = agent.categories.index(winning_category)
        data_envelope['category'] = winning_category

        return self.on_success  # (winning_category=winning_category_index)


class SpeakerPickMostConnectedWord(GuessingGameAction):
    """3. The speaker searches for words f in DS which are associated with cS , i.e.
        such that LS (f, cS ) > 0.
        If no associated words are found (i.e., LS (f, cS ) = 0 for all f ∈ DS ) or
        LS is empty, the speaker creates a new word f (i.e., DS := DS ∪ {f }), sets
        LS (f, cS ) = 0.5 and utters f .
        Now suppose that some associated words are found. Let f1,..., fn be all
        words associated with cS . The speaker chooses f from f1,..., fn such that
        LS (f, cS ) ≥ LS (fi, cS ), for i = 1, 2,..., n, and conveys f to the hearer. The
        choice of such a word is consistent with the concept of pragmatic meaning,
        defined later in this section."""

    def __init__(self, select_word_for_category: str):
        self.select_word_for_category = select_word_for_category

    def __call__(self, speaker: NewAgent, context, topic, data_envelope) -> str:
        assert data_envelope['category'], 'category must be defined on this stage'
        category = data_envelope['category']

        word = speaker.get_most_connected_word(category)
        # if not speaker.lexicon or all(v == 0.0 for v in speaker.lxc.get_row_by_col(category)):
        if word is None:
            logging.debug("%s(%d) introduces new word \"%s\"" % (speaker, speaker.agent_id, new_word))
            logging.debug("%s(%d) associates \"%s\" with his category" % (speaker, speaker.agent_id, new_word))
            word = new_word()
            speaker.add_new_word(word)
            speaker.learn_word_category(word, category)

        logging.debug("Speaker(%d) says: %s" % (speaker.agent_id, word))
        data_envelope['speaker_selected_word'] = word
        return self.select_word_for_category


class HearerGetMostConnectedHearerCategory(GuessingGameAction):
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

    def __init__(self, on_unknown_word: str, on_known_word: str):
        self.on_uknown_word = on_unknown_word
        self.on_known_word = on_known_word

    def __call__(self, hearer, context, topic, data_envelope):
        word = data_envelope['word']

        if not hearer.knows_word(word):
            return self.on_uknown_word

        category = hearer.get_most_connected_category(word)

        # if category is None:
        # todo
        # return self.on_uknown_word

        data_envelope['category'] = category

        return self.on_known_word


class HearerGetTopicAction(GuessingGameAction):
    """ 5. The topic is revealed to the hearer. If qS = qH , i.e., the topic is the same as
        the guess of the hearer, the game is successful. Otherwise, the game fails """

    def __init__(self, on_success: str, on_failure: str):
        self.on_success = on_success
        self.on_failure = on_failure

    def __call__(self, hearer: NewAgent, context, topic, data_envelope) -> str:
        assert 'category' in data_envelope.keys()
        category = data_envelope['category']

        selected = category.select(context)

        data_envelope['topic'] = topic

        if selected == topic:
            return self.on_success
        else:
            return self.on_failure


class OnComplete(GuessingGameAction):

    def __call__(self, speaker, hearer, context, topic, **kwargs) -> str:
        hearer_selected_topic = kwargs['topic']
        if hearer_selected_topic == topic:
            speaker_word = kwargs['word']
            speaker_category = kwargs['category']
            hearer_category = kwargs['hearer_category']
            # if self.completed and success1:
            speaker.update_on_success(speaker_word, speaker_category)
            hearer.update_on_success(speaker_word, hearer_category)
            # elif self.completed:
            #     hearer.update_on_failure(speaker_word, hearer_category)
            #     speaker.update_on_failure(speaker_word, speaker_category)
            return #GuessingGameStage.SUCCESS()
        else:
            return #GuessingGameStage.FAILURE()


class SuccessAction(GuessingGameAction):

    def __call__(self, speaker, hearer, context, topic, **kwargs) -> str:
        logging.debug("guessing game 1 success!")
        success1 = True
        # todo !!!
        # speaker.store_cs1_result(success1)
        # hearer.store_cs1_result(success1)
        # todo !!!
        hearer.update_on_success(speaker_word, hearer_category)
        speaker.update_on_success(speaker_word, speaker_category)
        return None


class FailureAction(GuessingGameAction):
    def __call__(self, speaker, hearer, context, topic, **kwargs) -> str:
        success1 = False
        logging.debug("guessing game 1 failed!")
        # todo
        # speaker.store_cs1_result(success1)
        # hearer.store_cs1_result(success1)
        # todo !!!
        # hearer.update_on_failure(speaker_word, hearer_category)
        # speaker.update_on_failure(speaker_word, speaker_category)
        return None


# class on_NO_CATEGORY_HEARER(GuessingGameAction):
#     def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
#         # def on_NO_CATEGORY(self, agent, context, topic):
#         return on_NO_DISCRIMINATION_HEARER()(speaker, hearer, context, topic)
#
#
# class on_NO_CATEGORY_SPEAKER(GuessingGameAction):
#     def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
#         # def on_NO_CATEGORY(self, agent, context, topic):
#         return on_NO_DISCRIMINATION_SPEAKER()(speaker, hearer, context, topic)


class on_NO_NOTICEABLE_DIFFERENCE(GuessingGameAction):
    def __call__(self, speaker, hearer, context, topic, **kwargs) -> str:
        logging.debug("no noticeable difference")
        return None


class on_NO_DISCRIMINATION_ACTION(GuessingGameAction):
    def __init__(self, select_agent):
        self.select_agent = select_agent

    def __call__(self, speaker, hearer, context, topic, data_envelope) -> str:
        agent = self.select_agent(speaker, hearer)
        logging.debug("no discrimination")
        logging.debug("%s(%d)" % (agent, agent.agent_id))
        agent.learn_stimulus(context[topic])
        return None


# class on_NO_DISCRIMINATION_HEARER(GuessingGameAction):
#     def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
#         return on_NO_DISCRIMINATION_ACTION()(hearer, context, topic)
#
#
# class on_NO_DISCRIMINATION_SPEAKER(GuessingGameAction):
#     def __call__(self, speaker, hearer, context, topic, **kwargs) -> GuessingGameStage:
#         return on_NO_DISCRIMINATION_ACTION()(speaker, context, topic)

# to be move to Speaker subclass


class on_NO_WORD_FOR_CATEGORY_SPEAKER(GuessingGameAction):
    def __call__(self, speaker, hearer, context, topic, data_envelope) -> str:
        agent_category = data_envelope["agent_category"]
        logging.debug("%s(%d) has no word for his category" % (speaker, speaker.agent_id))

        new_word = new_word()

        speaker.add_new_word(new_word)
        # TODO speaker_word instead new_word_index?
        logging.debug("%s(%d) introduces new word \"%s\"" % (speaker, speaker.agent_id, new_word))

        speaker.learn_word_category(new_word, agent_category)
        logging.debug("%s(%d) associates \"%s\" with his category" % (speaker, speaker.agent_id, new_word))
        return None


# class on_NO_SUCH_WORD_HEARER_OR_NO_ASSOCIATED_CATEGORIES(GuessingGameAction):
#     def __init__(self, discriminate_game: DiscriminationGameAction):
#         self.discriminate_game = discriminate_game
#
#     def __call__(self, speaker, hearer, context, topic, data_envelope) -> GuessingGameStage:
#         speaker_word: NewWord = data_envelope["speaker_word"]
#
#         logging.debug('on no such word event Hearer(%d) adds word "{}"'.format(hearer, speaker_word))
#         hearer.add_new_word(speaker_word)
#         logging.debug("{} plays the discrimination game".format(hearer))
#         # try:
#         # category = hearer.discrimination_game(context, topic)
#         next = self.discriminate_game(speaker, hearer, context, topic)
#         # logging.debug("Hearer(%d) category %d" % (hearer.id,
#         #                                           -1 if next is None else hearer.get_categories()[next].id))
#         # logging.debug("Hearer(%d) associates \"%s\" with his category %d" % (
#         #     hearer.agent_id, speaker_word, hearer.get_categories()[next].id))
#         hearer.learn_word_category(speaker_word, next)
#
#         return next


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='quantifiers simulation')

    # parser.add_argument('--simulation_name', '-sn', help='simulation name', type=str, default='test')
    parser.add_argument('--population_size', '-p', help='population size', type=int, default=10)
    parser.add_argument('--stimulus', '-stm', help='quotient or numeric', type=str, default='quotient',
                        choices=['quotient', 'numeric'])
    parser.add_argument('--max_num', '-mn', help='max number for numerics or max denominator for quotients', type=int,
                        default=100)
    parser.add_argument('--discriminative_threshold', '-dt', help='discriminative threshold', type=float, default=.95)
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
    load_inmemory_calculus(parsed_params['in_mem_calculus_path'], parsed_params['stimulus'])

    stimulus_factory = None
    if parsed_params['stimulus'] == 'quotient':
        stimulus_factory = QuotientBasedStimulusFactory(inmem['STIMULUS_LIST'], parsed_params['max_num'])
    if parsed_params['stimulus'] == 'numeric':
        stimulus_factory = NumericBasedStimulusFactory(inmem['STIMULUS_LIST'], parsed_params['max_num'])
    context_constructor = ContextFactory(stimulus_factory)

    population_size = 2  # parsed_params['population_size']
    steps = 10  # parsed_params['steps']
    seed = 0  # parsed_params['seed']

    game_params = GameParams(**parsed_params)

    population = NewPopulation(population_size, steps, seed)
    # game_graph = graphviz.Digraph()

    game_graph = {'2_SPEAKER_DISCRIMINATION_GAME':
        {'action': DiscriminationGameAction(
            on_no_category='SPEAKER_NO_CATEGORY_AFTER_DISCRIMINATION_GAME',
            on_no_noticeable_difference='SPEAKER_NO_CATEGORY_AFTER_DISCRIMINATION_GAME',
            on_no_discrimination='SPEAKER_NO_CATEGORY_AFTER_DISCRIMINATION_GAME',
            on_success='3_SPEAKER_PICKUPS_WORD_FOR_CATEGORY',
        ), 'agent': 'SPEAKER'},
        '3_SPEAKER_PICKUPS_WORD_FOR_CATEGORY':
            {'action': SpeakerPickMostConnectedWord(
                select_word_for_category='4_SPEAKER_CONVEYS_WORD_TO_THE_HEARER'
            ), 'agent': 'SPEAKER'},
        '4_SPEAKER_CONVEYS_WORD_TO_THE_HEARER':
            {'action': HearerGetMostConnectedHearerCategory(
                on_unknown_word='4_1_SPEAKER_CONVEYS_WORD_TO_THE_HEARER_ON_UNKNOWN_WORD',
                on_known_word='5_CHECK_TOPIC'),
                'agent': 'HEARER'},
        '4_1_SPEAKER_CONVEYS_WORD_TO_THE_HEARER_ON_UNKNOWN_WORD':
            {'action': DiscriminationGameAction(None, None, None, None),
             'agent': 'HEARER'},
        '5_CHECK_TOPIC':
            {'action': HearerGetTopicAction(on_success='SUCCESS', on_failure='FAILURE'),
             'agent': 'HEARER'}
    }

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

    for step in range(steps):
        paired_agents = population.select_pairs()
        for speaker, hearer in paired_agents:
            context = context_constructor()
            topic = 0
            logging.debug("Stimulus 1: %s" % context[0])
            logging.debug("Stimulus 2: %s" % context[1])
            logging.debug("topic = %d" % (topic + 1))

            state = game_graph['2_SPEAKER_DISCRIMINATION_GAME']
            action = state['action']
            select_agent = state['agent']
            data_envelope = {}
            while 'COMPLETE' not in state.keys():
                agent_selector = {'SPEAKER': select_speaker, 'HEARER': select_hearer}[select_agent]
                agent = agent_selector(speaker, hearer)
                next_state = action(agent, context, topic, data_envelope)
                state = game_graph[next_state]
                action = state['action']
                select_agent = state['agent']
