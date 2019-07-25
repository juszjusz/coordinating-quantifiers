from __future__ import division  # force python 3 division in python 2
import logging

from guessing_game_exceptions import NO_DIFFERENCE_FOR_CATEGORY, ERROR
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
        self.communicative_success = 0
        self.cs_scores = deque([0])

    def store_cs_result(self, result):
        if len(self.cs_scores) == 50:
            self.cs_scores.rotate(-1)
            self.cs_scores[-1] = int(result)
        else:
            self.cs_scores.append(int(result))

    def learn_word_category(self, word, category_index):
        self.initialize_word2category_connection(self.lexicon.index(word), category_index)

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
            raise ERROR

        category = self.categories[category]
        topic = category.select(context)
        if topic is None:
            raise NO_DIFFERENCE_FOR_CATEGORY
        return topic

    def update(self, success, role, word, category):
        i = self.lexicon.index(word)
        c = category
        if success and role == self.Role.SPEAKER:
            self.increment_word2category_connection(i, c)
            for k in range(len(self.categories)):
                if k != c:
                    self.decrement_word2category_connection(i, k)

        elif success and role == self.Role.HEARER:
            self.increment_word2category_connection(i, c)
            for j in range(len(self.lexicon)):
                if j != i:
                    self.decrement_word2category_connection(j, c)

        elif not success:
            self.decrement_word2category_connection(i, c)

    # HEARER: The hearer computes the cardinalities ... of word forms ... defined as ... (STAGE 7)
    def select_word(self, category):
        threshold = 0
        words_by_category = self.get_words(category=category)

        if len(words_by_category) == 1:
            return words_by_category[0]

        word1, word2 = words_by_category[0], words_by_category[1]
        categories1 = self.get_categories_above_threshold(word=word1, threshold=threshold)
        categories2 = self.get_categories_above_threshold(word=word2, threshold=threshold)
        return word1 if len(categories1) > len(categories2) else word2
