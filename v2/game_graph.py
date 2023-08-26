import logging
from typing import Tuple, Dict, Callable, List

import graphviz

from calculator import Calculator, NewAbstractStimulus
from domain_objects import NewCategory, NewWord, NewAgent, ThreadSafeWordFactory, AggregatedGameResultStats
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class GuessingGameAction:
    def __call__(self,
                 stats: AggregatedGameResultStats,
                 calculator: Calculator,
                 agent: NewAgent,
                 context: Tuple[NewAbstractStimulus, NewAbstractStimulus],
                 data_envelope: Dict,
                 **kwargs) -> str:
        pass

    def output_nodes(self) -> List[str]:
        return []

    def action_description(self) -> str:
        pass


class DiscriminationGameAction(GuessingGameAction):

    def __init__(self, on_no_category: str,
                 on_no_discrimination: str,
                 on_success: str,
                 selected_category_path: str):
        self.on_no_category = on_no_category
        self.on_no_discrimination = on_no_discrimination
        self.on_success = on_success

        self.selected_category_path = selected_category_path

    def output_nodes(self) -> List[str]:
        return [self.on_no_category, self.on_no_discrimination, self.on_success]

    def __call__(self, stats, calculator, agent: NewAgent, context, data_envelope: Dict, topic: int) -> str:
        if not agent.has_categories():
            logger.debug('no category {}({})'.format(agent, agent.agent_id))
            agent.learn_stimulus(context[topic], calculator)
            agent.add_discriminative_failure()
            return self.on_no_category

        s1, s2 = context

        assert s1.is_noticeably_different_from(s2), 'stimuli must be noticeably different'

        category1 = agent.get_best_matching_category(s1, calculator)
        category2 = agent.get_best_matching_category(s2, calculator)

        if category1 == category2:
            # logger.debug('no category {}({})'.format(agent, agent.agent_id))
            agent.learn_stimulus(context[topic], calculator)
            agent.add_discriminative_failure()
            return self.on_no_discrimination

        winning_category = [category1, category2][topic]

        # todo można to wbić do jednej metody, dwie metody występują tylko tutaj
        agent.reinforce_category(winning_category, context[topic], calculator)
        agent.forget_categories(winning_category)

        data_envelope[self.selected_category_path] = winning_category

        agent.add_discrimination_success()
        return self.on_success

    def action_description(self) -> str:
        return 'discrimination game'


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
                 on_no_word_for_category: str,
                 selected_word_path: str,
                 new_word: Callable[[], NewWord]):
        self._select_word_for_category = on_success
        self._on_no_word_for_category = on_no_word_for_category
        self._selected_word_path = selected_word_path
        self._new_word = new_word

    def __call__(self, stats, calculator, agent: NewAgent, context, data_envelope: Dict, category: NewCategory) -> str:
        word = agent.get_most_connected_word(category)

        if word is None:
            logger.debug("%s(%d) introduces new word \"%s\"" % (agent, agent.agent_id, word))
            logger.debug("%s(%d) associates \"%s\" with his category" % (agent, agent.agent_id, word))
            word = self._new_word()
            agent.add_new_word(word)
            agent.learn_word_category(word, category)
            return self._on_no_word_for_category
        else:
            logger.debug("Agent(%d) says: %s" % (agent.agent_id, word))
            data_envelope[self._selected_word_path] = word
            return self._select_word_for_category

    def output_nodes(self) -> List[str]:
        return [self._select_word_for_category]

    def action_description(self) -> str:
        return 'pick most connected word'


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

    def __init__(self, on_unknown_word_or_no_associated_category: str, on_success: str, selected_category_path: str):
        self._on_unknown_word_or_no_associated_category = on_unknown_word_or_no_associated_category
        self._on_known_word = on_success
        self._selected_category_path = selected_category_path

    def __call__(self, stats, calculator, agent: NewAgent, context, data_envelope: Dict, word: NewWord):
        if not agent.knows_word(word):
            agent.add_new_word(word)
            return self._on_unknown_word_or_no_associated_category

        category = agent.get_most_connected_category(word)

        if category is None:
            return self._on_unknown_word_or_no_associated_category

        data_envelope[self._selected_category_path] = category

        return self._on_known_word

    def output_nodes(self) -> List[str]:
        return [self._on_unknown_word_or_no_associated_category, self._on_known_word]

    def action_description(self) -> str:
        return 'pick most connected category'


class SelectAndCompareTopic(GuessingGameAction):
    """ 5. The topic is revealed to the hearer. If qS = qH , i.e., the topic is the same as
        the guess of the hearer, the game is successful. Otherwise, the game fails """

    def __init__(self, on_success: str, on_failure: str, flip_a_coin: Callable[[], int]):
        self._on_success = on_success
        self._on_failure = on_failure

        self._flip_a_coin = flip_a_coin

    def __call__(self, stats, calculator, agent: NewAgent, context, data_envelope: Dict, category: NewCategory,
                 topic: int) -> str:
        selected = category.select(context, calculator)

        if selected is None:
            selected = self._flip_a_coin()

        if selected == topic:
            return self._on_success
        else:
            return self._on_failure

    def output_nodes(self) -> List[str]:
        return [self._on_success, self._on_failure]

    def action_description(self) -> str:
        return 'select and compare topic'


class CompareWordsAction(GuessingGameAction):
    def __init__(self, on_equal_words: str, on_different_words: str):
        self._on_equal_words = on_equal_words
        self._on_different_words = on_different_words

    def __call__(self, stats, calculator, agent: NewAgent, context, data_envelope: Dict, speaker_word: NewWord,
                 hearer_word: NewWord) -> str:
        if speaker_word == hearer_word:
            return self._on_equal_words
        else:
            return self._on_different_words

    def output_nodes(self) -> List[str]:
        return [self._on_equal_words, self._on_different_words]

    def action_description(self) -> str:
        return 'compare words action'


class IncrementWordCategoryAssociation(GuessingGameAction):
    def __init__(self, on_success: str):
        self._on_success = on_success

    def __call__(self, stats, calculator, agent: NewAgent, context, data_envelope: Dict, word: NewWord,
                 category: NewCategory) -> str:
        agent.update_on_success(word, category)
        return self._on_success

    def output_nodes(self) -> List[str]:
        return [self._on_success]

    def action_description(self) -> str:
        return '+WxC association'


class DecrementWordCategoryAssociation(GuessingGameAction):
    def __init__(self, on_success: str):
        self._next = on_success

    def __call__(self, stats, calculator, agent: NewAgent, context, data_envelope: Dict, word: NewWord,
                 category: NewCategory) -> str:
        agent.update_on_failure(word, category)
        return self._next

    def output_nodes(self) -> List[str]:
        return [self._next]

    def action_description(self) -> str:
        return '-WxC association'


class SuccessAction(GuessingGameAction):
    def __init__(self, next: str):
        self._next = next

    def __call__(self, stats, calculator, agent: NewAgent, context, data_envelope: Dict) -> str:
        agent.add_communicative1_success()
        return self._next

    def output_nodes(self) -> List[str]:
        return [self._next]

    def action_description(self) -> str:
        return 'communicative1 success'


class FailureAction(GuessingGameAction):
    def __init__(self, next: str):
        self._next = next

    def __call__(self, stats, calculator, agent: NewAgent, context, data_envelope: Dict) -> str:
        agent.add_communicative1_failure()
        return self._next

    def output_nodes(self) -> List[str]:
        return [self._next]

    def action_description(self) -> str:
        return 'communicative1 failure'


class LearnWordCategoryAction(GuessingGameAction):
    def __init__(self, on_success: str):
        self._on_success = on_success

    def __call__(self, stats, calculator, agent: NewAgent, context, data_envelope: Dict, word: NewWord,
                 category: NewCategory) -> str:
        agent.learn_word_category(word, category)
        return self._on_success

    def output_nodes(self) -> List[str]:
        return [self._on_success]

    def action_description(self) -> str:
        return 'learn WxC association'


class CompleteAction(GuessingGameAction):
    def __init__(self, on_success: str):
        self._on_success = on_success

    def __call__(self, stats, calculator, agent: NewAgent, context, data_envelope: Dict) -> str:
        agent.update_discriminative_success_mean()
        agent.forget_words()
        return self._on_success

    def output_nodes(self) -> List[str]:
        return [self._on_success]

    def action_description(self) -> str:
        return 'forget words on complete'


class GameGraph:
    def __init__(self):
        self._graph = {}
        self._viz = graphviz.Digraph()
        self._start_node = []

    def add_simple_node(self, name: str, next_node: str, description: str, action: Callable[[NewAgent], None],
                        agent: str, args: List[str], is_start_node=False):
        pass

    def add_node(self, name: str, action: GuessingGameAction, agent: str, args: List[str], is_start_node=False):
        self._graph[name] = {'action': action, 'agent': agent, 'args': args}

        if action:
            if len(args):
                node_description = f'{name}\n{agent[:1]} {action.action_description()}\nexpects: {args}'
            else:
                node_description = f'{name}\n{agent[:1]} {action.action_description()}'
            self._viz.node(name, label=node_description)
            for out in set(action.output_nodes()):
                self._viz.edge(name, out)

        if is_start_node:
            self._start_node.append(name)

    def __call__(self, state: str):
        node = self._graph[state]
        agent = node['agent']
        action = node['action']
        args = node['args']
        return action, agent, args

    def viz(self):
        return self._viz

    def start(self):
        assert len(self._start_node) == 1, 'there must be a unique start node to start computation'
        return self(self._start_node[0])


def game_graph(flip_a_coin: Callable[[], int]) -> GameGraph:
    new_word = ThreadSafeWordFactory()

    g = GameGraph()
    g.add_node(name='2_SPEAKER_DISCRIMINATION_GAME',
               action=DiscriminationGameAction(
                   on_no_category='SPEAKER_COMPLETE_WITH_FAILURE',
                   on_no_discrimination='SPEAKER_COMPLETE_WITH_FAILURE',
                   on_success='3_SPEAKER_PICKUPS_WORD_FOR_CATEGORY',
                   selected_category_path='SPEAKER.category'
               ), agent='SPEAKER', args=['topic'], is_start_node=True)

    g.add_node(name='3_SPEAKER_PICKUPS_WORD_FOR_CATEGORY',
               action=PickMostConnectedWord(
                   on_success='4_SPEAKER_CONVEYS_WORD_TO_THE_HEARER',
                   on_no_word_for_category='SPEAKER_COMPLETE_WITH_FAILURE',
                   selected_word_path='SPEAKER.word',
                   new_word=new_word
               ), agent='SPEAKER', args=['SPEAKER.category'])

    g.add_node('4_SPEAKER_CONVEYS_WORD_TO_THE_HEARER',
               PickMostConnectedCategoryAction(
                   on_unknown_word_or_no_associated_category='4_1_SPEAKER_CONVEYS_WORD_TO_THE_HEARER_ON_UNKNOWN_WORD',
                   on_success='5_HEARER_CHECKS_TOPIC',
                   selected_category_path='HEARER.category'
               ), agent='HEARER', args=['SPEAKER.word'])

    g.add_node(name='4_1_SPEAKER_CONVEYS_WORD_TO_THE_HEARER_ON_UNKNOWN_WORD',
               action=DiscriminationGameAction(
                   on_no_category='SPEAKER_COMPLETE_WITH_FAILURE',
                   on_no_discrimination='SPEAKER_COMPLETE_WITH_FAILURE',
                   on_success='4_2_HEARER_LEARNS_WORD_CATEGORY',
                   selected_category_path='HEARER.category'
               ), agent='HEARER', args=['topic']),

    g.add_node(name='4_2_HEARER_LEARNS_WORD_CATEGORY',
               action=LearnWordCategoryAction(on_success='SPEAKER_COMPLETE_WITH_FAILURE'),
               agent='HEARER', args=['SPEAKER.word', 'HEARER.category']),

    g.add_node(name='5_HEARER_CHECKS_TOPIC',
               action=SelectAndCompareTopic(on_success='SPEAKER_SUCCESS',
                                            on_failure='SPEAKER_FAILURE',
                                            flip_a_coin=flip_a_coin),
               agent='HEARER', args=['HEARER.category', 'topic']),

    g.add_node(name='SPEAKER_SUCCESS',
               action=IncrementWordCategoryAssociation(on_success='HEARER_SUCCESS'),
               agent='SPEAKER', args=['SPEAKER.word', 'SPEAKER.category']),

    g.add_node(name='HEARER_SUCCESS',
               action=IncrementWordCategoryAssociation(on_success='SPEAKER_COMPLETE_WITH_SUCCESS'),
               agent='HEARER', args=['SPEAKER.word', 'HEARER.category']),

    g.add_node(name='SPEAKER_FAILURE',
               action=DecrementWordCategoryAssociation(on_success='HEARER_FAILURE'),
               agent='SPEAKER', args=['SPEAKER.word', 'SPEAKER.category']),

    g.add_node(name='HEARER_FAILURE',
               action=DecrementWordCategoryAssociation(on_success='SPEAKER_COMPLETE_WITH_FAILURE'),
               agent='HEARER', args=['SPEAKER.word', 'HEARER.category']),

    g.add_node(name='SPEAKER_COMPLETE_WITH_SUCCESS',
               action=SuccessAction(next='HEARER_COMPLETE_WITH_SUCCESS'), agent='SPEAKER', args=[]),

    g.add_node(name='HEARER_COMPLETE_WITH_SUCCESS',
               action=SuccessAction(next='SPEAKER_COMPLETE'), agent='HEARER', args=[]),

    g.add_node(name='SPEAKER_COMPLETE_WITH_FAILURE',
               action=FailureAction(next='HEARER_COMPLETE_WITH_FAILURE'), agent='SPEAKER', args=[]),

    g.add_node(name='HEARER_COMPLETE_WITH_FAILURE',
               action=FailureAction(next='SPEAKER_COMPLETE'), agent='HEARER', args=[]),

    g.add_node(name='SPEAKER_COMPLETE', action=CompleteAction(on_success='HEARER_COMPLETE'), agent='SPEAKER', args=[]),

    g.add_node(name='HEARER_COMPLETE', action=CompleteAction(on_success='NEXT_STEP'), agent='HEARER', args=[]),

    g.add_node(name='NEXT_STEP', action=None, agent=None, args=[])

    return g


def game_graph_with_stage_7(flip_a_coin: Callable[[], int]) -> GameGraph:
    new_word = ThreadSafeWordFactory()

    g = GameGraph()
    g.add_node(name='2_SPEAKER_DISCRIMINATION_GAME',
               action=DiscriminationGameAction(
                   on_no_category='SPEAKER_COMPLETE_WITH_FAILURE',
                   on_no_discrimination='SPEAKER_COMPLETE_WITH_FAILURE',
                   on_success='3_SPEAKER_PICKUPS_WORD_FOR_CATEGORY',
                   selected_category_path='SPEAKER.category'
               ), agent='SPEAKER', args=['topic'], is_start_node=True)

    g.add_node(name='3_SPEAKER_PICKUPS_WORD_FOR_CATEGORY',
               action=PickMostConnectedWord(
                   on_success='4_SPEAKER_CONVEYS_WORD_TO_THE_HEARER',
                   selected_word_path='SPEAKER.word',
                   new_word=new_word
               ), agent='SPEAKER', args=['SPEAKER.category'])

    g.add_node('4_SPEAKER_CONVEYS_WORD_TO_THE_HEARER',
               PickMostConnectedCategoryAction(
                   on_unknown_word_or_no_associated_category='4_1_SPEAKER_CONVEYS_WORD_TO_THE_HEARER_ON_UNKNOWN_WORD',
                   on_success='5_HEARER_CHECKS_TOPIC',
                   selected_category_path='HEARER.category'
               ), agent='HEARER', args=['SPEAKER.word']
               )

    g.add_node(name='4_1_SPEAKER_CONVEYS_WORD_TO_THE_HEARER_ON_UNKNOWN_WORD',
               action=DiscriminationGameAction(
                   on_no_category='SPEAKER_COMPLETE_WITH_FAILURE',
                   on_no_discrimination='SPEAKER_COMPLETE_WITH_FAILURE',
                   on_success='4_2_HEARER_LEARNS_WORD_CATEGORY',
                   selected_category_path='HEARER.category'
               ), agent='HEARER', args=['topic']),

    g.add_node(name='4_2_HEARER_LEARNS_WORD_CATEGORY',
               action=LearnWordCategoryAction(on_success='SPEAKER_COMPLETE_WITH_FAILURE'),
               agent='HEARER', args=['SPEAKER.word', 'HEARER.category']),

    g.add_node(name='5_HEARER_CHECKS_TOPIC',
               action=SelectAndCompareTopic(on_success='SPEAKER_SUCCESS',
                                            on_failure='TOPIC_GUESS_FAILURE',
                                            flip_a_coin=flip_a_coin),
               agent='HEARER', args=['HEARER.category', 'topic']),

    g.add_node(name='TOPIC_GUESS_FAILURE',
               action=DiscriminationGameAction(
                   on_no_category='SPEAKER_COMPLETE_WITH_FAILURE_7',
                   on_no_discrimination='SPEAKER_COMPLETE_WITH_FAILURE_7',
                   on_success='6_HEARER_PICKUPS_WORD_FOR_CATEGORY',
                   selected_category_path='HEARER.category'
               ), agent='HEARER', args=['topic']),

    g.add_node(name='SPEAKER_SUCCESS',
               action=IncrementWordCategoryAssociation(on_success='HEARER_SUCCESS'),
               agent='SPEAKER', args=['SPEAKER.word', 'SPEAKER.category']),

    g.add_node(name='HEARER_SUCCESS',
               action=IncrementWordCategoryAssociation(on_success='SPEAKER_COMPLETE_WITH_SUCCESS'),
               agent='HEARER', args=['SPEAKER.word', 'HEARER.category']),

    g.add_node(name='SPEAKER_SUCCESS_7',
               action=IncrementWordCategoryAssociation(on_success='HEARER_SUCCESS_7'),
               agent='SPEAKER', args=['SPEAKER.word', 'SPEAKER.category']),

    g.add_node(name='HEARER_SUCCESS_7',
               action=IncrementWordCategoryAssociation(on_success='SPEAKER_COMPLETE_WITH_SUCCESS_7'),
               agent='HEARER', args=['HEARER.word', 'HEARER.category']),

    g.add_node(name='SPEAKER_FAILURE',
               action=DecrementWordCategoryAssociation(on_success='HEARER_FAILURE'),
               agent='SPEAKER', args=['SPEAKER.word', 'SPEAKER.category']),

    g.add_node(name='HEARER_FAILURE',
               action=DecrementWordCategoryAssociation(on_success='SPEAKER_COMPLETE'),
               agent='HEARER', args=['SPEAKER.word', 'HEARER.category']),

    g.add_node(name='SPEAKER_COMPLETE_WITH_SUCCESS',
               action=SuccessAction(next='HEARER_COMPLETE_WITH_SUCCESS'), agent='SPEAKER', args=[]),

    g.add_node(name='HEARER_COMPLETE_WITH_SUCCESS',
               action=SuccessAction(next='SPEAKER_COMPLETE'), agent='HEARER', args=[]),

    g.add_node(name='SPEAKER_COMPLETE_WITH_SUCCESS_7',
               action=SuccessAction(next='HEARER_COMPLETE_WITH_SUCCESS_7'), agent='SPEAKER', args=[]),

    g.add_node(name='HEARER_COMPLETE_WITH_SUCCESS_7',
               action=SuccessAction(next='SPEAKER_COMPLETE'), agent='HEARER', args=[]),

    g.add_node(name='SPEAKER_COMPLETE_WITH_FAILURE',
               action=FailureAction(next='HEARER_COMPLETE_WITH_FAILURE'), agent='SPEAKER', args=[]),

    g.add_node(name='HEARER_COMPLETE_WITH_FAILURE',
               action=FailureAction(next='SPEAKER_COMPLETE'), agent='HEARER', args=[]),

    g.add_node(name='SPEAKER_COMPLETE_WITH_FAILURE_7',
               action=FailureAction(next='HEARER_COMPLETE_WITH_FAILURE_7'), agent='SPEAKER', args=[]),

    g.add_node(name='HEARER_COMPLETE_WITH_FAILURE_7',
               action=FailureAction(next='SPEAKER_COMPLETE'), agent='HEARER', args=[]),

    g.add_node(name='SPEAKER_COMPLETE', action=CompleteAction(on_success='HEARER_COMPLETE'), agent='SPEAKER', args=[]),
    g.add_node(name='HEARER_COMPLETE', action=CompleteAction(on_success='NEXT_STEP'), agent='HEARER', args=[]),

    g.add_node(name='NEXT_STEP', action=None, agent=None, args=[])

    return g


if __name__ == '__main__':
    # to render graph you need to have dot installed
    # https://graphviz.org
    g = game_graph(None)
    dot = g.viz()
    print('''you can also just copy paste encoded graph printed below 
    in graphviz online gui (https://dreampuf.github.io/GraphvizOnline):''')
    print(dot.source)
    dot.format = 'png'
    dot.render('game_graph', view=True)
