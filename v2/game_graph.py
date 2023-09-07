import logging
from copy import copy
from typing import Tuple, Dict, Callable, List

import graphviz
import networkx as nx

from domain_objects import NewCategory, NewWord, NewAgent, SimpleCounter
from calculator import Stimulus

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class GuessingGameAction:
    def __call__(self,
                 agent: NewAgent,
                 context: Tuple[Stimulus, Stimulus],
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

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, topic: int) -> str:
        if not agent.has_categories():
            logger.debug('no category {}({})'.format(agent, agent.agent_id))
            agent.learn_stimulus(context[topic])
            agent.add_discriminative_failure()
            return self.on_no_category

        s1, s2 = context

        category1 = agent.get_most_responsive_category(s1)
        category2 = agent.get_most_responsive_category(s2)

        if category1 == category2:
            # logger.debug('no category {}({})'.format(agent, agent.agent_id))
            agent.learn_stimulus(context[topic])
            agent.add_discriminative_failure()
            return self.on_no_discrimination

        winning_category = [category1, category2][topic]

        # todo można to wbić do jednej metody, dwie metody występują tylko tutaj
        agent.reinforce_category(winning_category, context[topic])
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
                 selected_word_path: str):
        self._select_word_for_category = on_success
        self._on_no_word_for_category = on_no_word_for_category
        self._selected_word_path = selected_word_path
        self._new_word_counter = SimpleCounter()

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, category: NewCategory) -> str:
        word = agent.get_most_connected_word(category)

        if word is None:
            logger.debug("%s(%d) introduces new word \"%s\"" % (agent, agent.agent_id, word))
            logger.debug("%s(%d) associates \"%s\" with his category" % (agent, agent.agent_id, word))
            id = self._new_word_counter()
            new_word = NewWord(word_id=id, originated_from_category=copy(category))
            agent.add_new_word(new_word)
            agent.learn_word_category(new_word, category)
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

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, word: NewWord):
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

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, category: NewCategory, topic: int) -> str:
        selected = agent.select_stimuli_by_category(category, context)

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

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, speaker_word: NewWord,
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

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, word: NewWord,
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

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, word: NewWord,
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

    def __call__(self, agent: NewAgent, context, data_envelope: Dict) -> str:
        agent.add_communicative1_success()
        return self._next

    def output_nodes(self) -> List[str]:
        return [self._next]

    def action_description(self) -> str:
        return 'communicative1 success'


class FailureAction(GuessingGameAction):
    def __init__(self, next: str):
        self._next = next

    def __call__(self, agent: NewAgent, context, data_envelope: Dict) -> str:
        agent.add_communicative1_failure()
        return self._next

    def output_nodes(self) -> List[str]:
        return [self._next]

    def action_description(self) -> str:
        return 'communicative1 failure'


class LearnWordCategoryAction(GuessingGameAction):
    def __init__(self, on_success: str):
        self._on_success = on_success

    def __call__(self, agent: NewAgent, context, data_envelope: Dict, word: NewWord,
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

    def __call__(self, agent: NewAgent, context, data_envelope: Dict) -> str:
        agent.update_discriminative_success_mean()
        agent.forget_words()
        return self._on_success

    def output_nodes(self) -> List[str]:
        return [self._on_success]

    def action_description(self) -> str:
        return 'forget words on complete'


class StartAction(GuessingGameAction):
    def __init__(self, start):
        self._on_start = start

    def __call__(self, agent: NewAgent, *args, **kwargs):
        agent.next_step()
        return self._on_start

    def output_nodes(self) -> List[str]:
        return [self._on_start]

    def action_description(self) -> str:
        return 'start'


class EmptyAction(GuessingGameAction):

    def output_nodes(self) -> List[str]:
        return []

    def action_description(self) -> str:
        return 'empy node'


class GameGraph:
    def __init__(self):
        self._graph = {}
        self._action = {}
        self._agent = {}
        self._args = {}
        self._viz = graphviz.Digraph()
        self._start_node = []

    def add_node(self, name: str, action: GuessingGameAction, agent: str, args: List[str], is_start_node=False):
        self._graph[name] = action.output_nodes()
        self._action[name] = action
        self._agent[name] = agent
        self._args[name] = args

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
        agent = self._agent[state]
        action = self._action[state]
        args = self._args[state]
        return action, agent, args

    def start(self):
        assert len(self._start_node) == 1, 'there must be a unique start node to start computation'
        start_node = self._start_node[0]
        return start_node, *self(start_node)

    def viz(self):
        return self._viz

    @staticmethod
    def map_to_nxGraph(G):
        return nx.DiGraph(G._graph)


def game_graph(flip_a_coin: Callable[[], int]) -> GameGraph:
    g = GameGraph()

    g.add_node(name='START_SPEAKER', action=StartAction(start='START_HEARER'),
               agent='SPEAKER', args=[], is_start_node=True)

    g.add_node(name='START_HEARER', action=StartAction(start='2_SPEAKER_DISCRIMINATION_GAME'),
               agent='HEARER', args=[])

    g.add_node(name='2_SPEAKER_DISCRIMINATION_GAME',
               action=DiscriminationGameAction(
                   on_no_category='SPEAKER_COMPLETE_WITH_FAILURE',
                   on_no_discrimination='SPEAKER_COMPLETE_WITH_FAILURE',
                   on_success='3_SPEAKER_PICKUPS_WORD_FOR_CATEGORY',
                   selected_category_path='SPEAKER.category'
               ), agent='SPEAKER', args=['topic'])

    g.add_node(name='3_SPEAKER_PICKUPS_WORD_FOR_CATEGORY',
               action=PickMostConnectedWord(
                   on_success='4_SPEAKER_CONVEYS_WORD_TO_THE_HEARER',
                   on_no_word_for_category='SPEAKER_COMPLETE_WITH_FAILURE',
                   selected_word_path='SPEAKER.word'
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

    g.add_node(name='NEXT_STEP', action=EmptyAction(), agent='', args=[])

    return g


def game_graph_with_stage_7(flip_a_coin: Callable[[], int]) -> GameGraph:
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
                   selected_word_path='SPEAKER.word'
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


def animate_graph_transition_frequency(states_edges_cnts_normalized):
    edge_labels_cnts = states_edges_cnts_normalized
    G: GameGraph = game_graph(None)
    nxG = GameGraph.map_to_nxGraph(G)
    edges = list(nxG.edges)
    nodes = list(nxG.nodes)
    # osage, acyclic, nop, neato, sfdp, patchwork, ccomps, unflatten, dot, gc, circo, gvpr, twopi, tred, gvcolor, fdp, sccmap.
    pos = nx.nx_agraph.pygraphviz_layout(nxG, prog='dot')
    fig, ax = plt.subplots(figsize=(15, 15))

    def update(num):
        ax.clear()

        frame, edge_labels = edge_labels_cnts[num]
        zero_edges = {e for e in edges if e not in edge_labels.keys()}
        for e in zero_edges:
            edge_labels[e] = 0
        nx.draw(nxG, pos, ax=ax, node_size=1000, with_labels=True,
                edge_cmap=edge_labels.values(),
                font_size=8,
                node_color='skyblue',
                labels={n: n for n in nodes})

        # Słownik wag dla krawędzi
        # edge_labels = {(edges[i][0], edges[i][1]): edge_weights[i] for i in range(len(edges))}

        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels)
        ax.legend(labels=['steps: ' + str(frame)])

    ani = animation.FuncAnimation(fig, update, frames=len(edge_labels_cnts), repeat=False)
    ani.save('animated_graph.gif', writer='pillow', fps=1)


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

    nxG = GameGraph.map_to_nxGraph(g)
    print(nxG)
