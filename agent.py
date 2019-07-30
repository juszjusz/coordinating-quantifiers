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
        self.agents = [Agent(agent_id, Language(), 0, deque([0])) for agent_id in range(population_size)]

    def select_pairs_per_round(self, games_per_round):
        agents_per_game = sample(self.agents, games_per_round * 2)
        return [(Speaker(a1), Hearer(a2)) for a1, a2 in zip(agents_per_game[::2], agents_per_game[1::2])]


class Agent:
    class Result:
        SUCCESS = 1
        FAILURE = 0

    def __init__(self, id, language, communicative_success, cs_scores):
        self.id = id
        self.language = language
        self.communicative_success = communicative_success
        self.cs_scores = cs_scores

    def store_cs_result(self, result):
        if len(self.cs_scores) == 50:
            self.cs_scores.rotate(-1)
            self.cs_scores[-1] = int(result)
        else:
            self.cs_scores.append(int(result))

    def learn_word_category(self, word, category_index):
        self.language.initialize_word2category_connection(word, category_index)

    def discrimination_game(self, context, topic):
        return self.language.discrimination_game(context, topic)

    def get_discriminative_success(self):
        return self.language.discriminative_success

    def get_most_connected_word(self, category):
        return self.language.get_most_connected_word(category)

    def get_most_connected_category(self, word):
        return self.language.get_most_connected_category(word)

    def get_categories(self):
        return self.language.categories

    def get_categories_by_word(self, word):
        return self.language.get_categories_by_word(word)

    def get_lexicon(self):
        return self.language.lexicon

    def get_words_by_category(self, category):
        return self.language.get_words_by_category(category)

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

    def learn_stimulus(self, context, n, category=None):
        logging.debug(" learns stimulus %d by " % (n + 1))
        if self.language.discriminative_success >= Perception.discriminative_threshold and category is not None:
            logging.debug("updating category")
            self.language.update_category(category, context[n])
            return category
        else:
            logging.debug("adding new category centered on %f" % (context[n].a / context[n].b))
            return self.language.add_category(context[n])

    def update_on_failure(self, word, category):
        self.language.decrement_word2category_connection(word, category)


class Speaker(Agent):
    def __init__(self, agent):
        Agent.__init__(self, agent.id, agent.language, agent.communicative_success, agent.cs_scores)

    def update_on_success(self, word, category):
        self.language.increment_word2category_connection(word=word, category_index=category)
        for k_index, _ in self.language.get_categories_sorted_by_val(word):
            if k_index != category:
                self.language.decrement_word2category_connection(word=word, category_index=k_index)

    def update_on_success_stage7(self, word, category):
        self.language.increment_word2category_connection(word=word, category_index=category)

    def add_new_word(self):
        return self.language.add_new_word()


class Hearer(Agent):
    def __init__(self, agent):
        Agent.__init__(self, agent.id, agent.language, agent.communicative_success, agent.cs_scores)

    def get_topic(self, context, category):
        if category is None:
            raise ERROR

        category = self.language.categories[category]
        topic = category.select(context)
        if topic is None:
            raise NO_DIFFERENCE_FOR_CATEGORY
        return topic

    # HEARER: The hearer computes the cardinalities ... of word forms ... defined as ... (STAGE 7)
    def select_word(self, category):
        threshold = .005  # todo
        words_by_category = self.language.get_words_sorted_by_val(category=category)

        word0 = words_by_category[0]
        categories0 = list(
            filter(lambda cat2propensity: cat2propensity[1] > threshold, self.language.get_categories_sorted_by_val(word0)))

        if len(words_by_category) == 1:
            return word0, categories0

        word1 = words_by_category[1]
        categories1 = list(
            filter(lambda cat2propensity: cat2propensity[1] > threshold, self.language.get_categories_sorted_by_val(word1)))

        return (word0, categories0) if len(categories0) > len(categories1) else (word1, categories1)

    def update_on_success(self, speaker_word, hearer_category):
        self.language.increment_word2category_connection(word=speaker_word, category_index=hearer_category)
        for v in self.language.get_words_sorted_by_val(hearer_category):
            if v != speaker_word:
                self.language.decrement_word2category_connection(word=v, category_index=hearer_category)

    def update_on_success_stage7(self, word, word_categories):
        for c_index, _ in word_categories:
            self.language.increment_word2category_connection(word, c_index)

    def add_word(self, word):
        return self.language.add_word(word)
