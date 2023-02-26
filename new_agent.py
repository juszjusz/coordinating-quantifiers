from __future__ import division  # force python 3 division in python 2
import logging

from scipy.sparse import coo_array

from guessing_game_exceptions import NO_DIFFERENCE_FOR_CATEGORY, ERROR
from language import Language
from random import sample
from collections import deque
from guessing_game_exceptions import NO_WORD_FOR_CATEGORY
from itertools import izip
from numpy import ndarray, asarray

from new_guessing_game import GuessingGameEvent, DiscriminationGameWordFound
from new_language import NewLanguage


class Population:

    def __init__(self, population_size):
        self.population_size = population_size
        self.agents = [NewAgent(agent_id, NewLanguage(params)) for agent_id in
                       range(self.population_size)]

    def select_pairs_per_round(self, games_per_round):
        agents_per_game = sample(self.agents, games_per_round * 2)
        return [(NewSpeaker(a1), NewHearer(a2)) for a1, a2 in zip(agents_per_game[::2], agents_per_game[1::2])]

    def __iter__(self):
        return iter(self.agents)

    def __len__(self):
        return len(self.agents)


class NewAgent:

    def __init__(self, agent_id: int, rounds: int):
        self.agent_id = agent_id
        self.categories = None
        self.lang_cats_association = coo_array((rounds, rounds), dtype=float)

    def create_word_category_association(self, category):
        self.lang_cats_association[0, 0] = 0


class NewSpeaker(NewAgent):

    def __init__(self, agent):
        NewAgent.__init__(self, agent.agent_id)

    def update_on_success(self, word, category):
        self.language.increment_word2category_connection(word=word, category_index=category)
        self.language.inhibit_word2categories_connections(word=word, category_index=category)

    def update_on_success2c(self, word, category):
        logging.debug("Incrementing connections for %s, agent %d" % (word, self.agent_id))
        csimilarities = [self.language.csimilarity(word, c) for c in self.language.categories]
        logging.debug("Speaker successful category %d, its similarity %f to %s meaning" % (
            category, csimilarities[category], word))
        logging.debug("Similarities: %s" % str(csimilarities))
        self.language.increment_word2category_connections_by_csimilarity(word, csimilarities)
        self.language.inhibit_word2categories_connections(word=word, category_index=category)

    def update_on_success_stage7(self, word, category):
        self.language.increment_word2category_connection(word=word, category_index=category)

    def add_new_word(self):
        return self.language.add_new_word()

    def find_discriminating_word(self, context, topic) -> GuessingGameEvent:
        # DISCRIMINATION GAME
        discriminating_category = self.__discrimination_game_old(context, topic)

        return DiscriminationGameWordFound(self.__get_most_connected_word(discriminating_category))

    def __discrimination_game_old(self, context, topic):
        winning_category = self.discriminate(context, topic)
        winning_category.reinforce(context[topic], self.beta)
        self.forget_categories(winning_category)
        return self.categories.index(winning_category)

    def __get_most_connected_word(self, category):
        return self.language.__get_most_connected_word(category)


class NewHearer(NewAgent):
    def __init__(self, agent):
        NewAgent.__init__(self, agent.agent_id)

    # HEARER: The hearer computes the cardinalities ... of word forms ... defined as ... (STAGE 7)
    def select_word(self, category):
        threshold = .000  # todo
        words_by_category = self.language.get_words_sorted_by_val(category=category)

        if not len(words_by_category):
            return None, None

        word0 = words_by_category[0]
        categories0 = list(
            filter(lambda cat2propensity: cat2propensity[1] > threshold,
                   self.language.get_categories_sorted_by_val(word0)))

        if len(words_by_category) == 1:
            return word0, categories0

        word1 = words_by_category[1]
        categories1 = list(
            filter(lambda cat2propensity: cat2propensity[1] > threshold,
                   self.language.get_categories_sorted_by_val(word1)))

        logging.debug(
            "Two words sorted by cardinality: %s, %s" % (word0, word1) if len(categories0) > len(categories1) else (
                word1, word0))
        return (word0, categories0) if len(categories0) > len(categories1) else (word1, categories1)

    def update_on_success(self, speaker_word, hearer_category):
        self.language.increment_word2category_connection(word=speaker_word, category_index=hearer_category)
        self.language.inhibit_category2words_connections(word=speaker_word, category_index=hearer_category)

    def update_on_success2c(self, word, category):
        logging.debug("Incrementing connections for %s, agent %d" % (word, self.agent_id))
        csimilarities = [self.language.csimilarity(word, c) for c in self.language.categories]
        logging.debug("Hearer successful category %d, its similarity %f to %s meaning" % (
            self.get_categories()[category].agent_id, csimilarities[category], word))
        logging.debug("c Similarities: %s" % str(csimilarities))
        self.language.increment_word2category_connections_by_csimilarity(word, csimilarities)
        self.language.inhibit_category2words_connections(word=word, category_index=category)

    def update_on_success_stage7(self, word, word_categories):
        for c_index, _ in word_categories:
            self.language.increment_word2category_connection(word, c_index)

    def find_topic(self, context, speaker_word):
        return self.get_most_connected_category(speaker_word)
