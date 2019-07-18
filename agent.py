from __future__ import division  # force python 3 division in python 2
import logging
from language import Language
from language import Perception
from random import sample
from collections import deque
import matplotlib.pyplot as plt
from numpy import linspace


class Population:

    def __init__(self, population_size):
        self.population_size = population_size
        self.agents = [Agent(agent_id) for agent_id in range(population_size)]

    def select_pairs_per_round(self, games_per_round):
        agents_per_game = sample(self.agents, games_per_round * 2)
        return [(a1, a2) for a1, a2 in zip(agents_per_game[::2], agents_per_game[1::2])]


class Agent(Language):
    class Result:
        SUCCESS = 1
        FAILURE = 0

    class Role:
        SPEAKER = 1
        HEARER = 2

    def __init__(self, id):
        Language.__init__(self)
        self.id = id
        self.discriminative_success = 0
        self.communicative_success = 0
        self.ds_scores = deque([0])
        self.cs_scores = deque([0])

    def store_ds_result(self, result):
        if len(self.ds_scores) == 50:
            self.ds_scores.rotate(-1)
            self.ds_scores[-1] = int(result)
        else:
            self.ds_scores.append(int(result))

    def store_cs_result(self, result):
        if len(self.cs_scores) == 50:
            self.cs_scores.rotate(-1)
            self.cs_scores[-1] = int(result)
        else:
            self.cs_scores.append(int(result))

    def learn_word_category(self, word, category_index):
        self.lxc[self.lexicon.index(word), category_index] = 0.5

    # def learn_word_topic(self, word: str, context: list, topic: int):
    #     c_j = self.discriminate(context, topic)
    #     w_i = len(self.lexicon)
    #     self.lexicon.append(word)
    #     rows_cnt, cols_cnt = self.lxc.shape
    #     self.lxc.resize((rows_cnt + 1, cols_cnt), refcheck=False)
    #     if c_j is None:
    #         # TODO question: is this ok or maybe learn_topic?
    #         return
    #     else:

    def learn_stimulus(self, category, context, n):
        logging.debug(" learns stimulus %d by " % (n + 1))
        if self.discriminative_success >= Perception.discriminative_threshold and category is not None:
            logging.debug("updating category")
            self.update_category(category, context[n])
            return category
        else:
            logging.debug("adding new category centered on %f" % (context[n].a / context[n].b))
            return self.add_category(context[n])

    def get_topic(self, context, category):
        if category is None:
            return None, Language.Error.ERROR

        category = self.categories[category]
        topic = category.select(context)
        return (topic, Perception.Error.NO_DIFFERENCE_FOR_CATEGORY) if topic is None \
            else (topic, Perception.Error.NO_ERROR)

    def update(self, success, role, word, category):
        i = self.lexicon.index(word)
        c = category
        if success and role == self.Role.SPEAKER:
            self.lxc[i, c] = self.lxc[i, c] + 0.1 * self.lxc[i, c]
            for k in range(len(self.categories)):
                if k != c:
                    self.lxc[i, k] = self.lxc[i, k] - 0.1 * self.lxc[i, k]

        elif success and role == self.Role.HEARER:
            self.lxc[i, c] = self.lxc[i, c] + 0.1 * self.lxc[i, c]
            for j in range(len(self.lexicon)):
                if j != i:
                    self.lxc[j, c] = self.lxc[j, c] - 0.1 * self.lxc[j, c]

        elif not success:
            self.lxc[i, c] = self.lxc[i, c] - 0.1 * self.lxc[i, c]

# class Speaker(Agent):
#     def __init__(self, agent):
#         super().__init__()
#         Agent.__init__(self, id)
#
#
# class Hearer(Agent):
#     def __init__(self, id):
#         Agent.__init__(self, id)