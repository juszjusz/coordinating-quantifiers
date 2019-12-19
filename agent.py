from __future__ import division  # force python 3 division in python 2
import logging

from guessing_game_exceptions import NO_DIFFERENCE_FOR_CATEGORY, ERROR
from language import Language
from random import sample
from collections import deque
from guessing_game_exceptions import NO_WORD_FOR_CATEGORY
import stimulus
from itertools import izip
from numpy import array


class Population:

    def __init__(self, params):
        self.population_size = params['population_size']
        self.agents = [Agent(agent_id, Language(params), deque([0]), deque([0]), deque([0])) for agent_id in range(self.population_size)]
        self.ds = []
        self.cs1 = []
        self.cs2 = []
        self.cs12 = []

    def select_pairs_per_round(self, games_per_round):
        agents_per_game = sample(self.agents, games_per_round * 2)
        return [(Speaker(a1), Hearer(a2)) for a1, a2 in zip(agents_per_game[::2], agents_per_game[1::2])]

    def __iter__(self):
        return iter(self.agents)

    def __len__(self):
        return len(self.agents)

    def update_metrics(self):
        self.ds.append(sum(map(lambda agent: agent.get_discriminative_success() * 100.0, self.agents)) / len(self.agents))
        self.cs1.append(sum(map(lambda agent: agent.get_communicative_success() * 100.0, self.agents)) / len(self.agents))
        self.cs2.append(sum(map(lambda agent: agent.get_communicative_success2() * 100.0, self.agents)) / len(self.agents))
        self.cs12.append(sum(map(lambda agent: agent.get_communicative_success12() * 100.0, self.agents)) / len(self.agents))

    #def update_ds(self):
    #    self.ds.append(sum((map(lambda agent: agent.get_discriminative_success() * 100, self.agents))) / len(self.agents))

    #def update_cs(self):
    #    self.cs.append(sum(map(Agent.get_communicative_success, self.agents)) / len(self.agents))
    #    self.cs.append(sum(map(Agent.get_communicative_success2, self.agents)) / len(self.agents))

    def get_mon(self, stimuluses):
       return sum(map(lambda agent: agent.get_monotonicity(stimuluses) * 100.0, self.agents)) / len(self.agents)

    def get_convexity(self, stimuluses):
        return sum(map(lambda agent: agent.get_convexity(stimuluses) * 100.0, self.agents)) / len(self.agents)


class Agent:

    def __init__(self, id, language, cs1_scores, cs2_scores, cs12_scores):
        self.id = id
        self.language = language
        self.cs1_scores = cs1_scores
        self.cs2_scores = cs2_scores
        self.cs12_scores = cs12_scores

    def store_cs1_result(self, result):
        self.__store_result__(result, self.cs1_scores)

    def store_cs12_result(self, result):
        self.__store_result__(result, self.cs12_scores)

    def store_cs2_result(self, result):
        if result is not None:
            self.__store_result__(result, self.cs2_scores)

    def __store_result__(self, result, history):
        if len(history) == 50:
            history.rotate(-1)
            history[-1] = result
        else:
            history.append(result)

    def learn_word_category(self, word, category_index):
        self.language.initialize_word2category_connection(word, category_index)

    def discrimination_game(self, context, topic):
        return self.language.discrimination_game(context, topic)

    def get_discriminative_success(self):
        return self.language.discriminative_success

    def get_communicative_success(self):
        return sum(self.cs1_scores) / len(self.cs1_scores)

    def get_communicative_success2(self):
        return sum(self.cs2_scores) / len(self.cs2_scores)

    def get_communicative_success12(self):
        return sum(self.cs12_scores) / len(self.cs12_scores)

    def get_best_matching_words(self, stimuluses):
        words = [self.get_best_matching_word(s) for s in stimuluses]
        return words

    def get_best_matching_word(self, stimulus):
        category = self.get_best_matching_category(stimulus)
        if category is None:
            return "?"
        else:
            try:
                word = self.get_most_connected_word(category)
            except NO_WORD_FOR_CATEGORY:
                return "?"
            return word

    def pragmatic_meaning(self, word, stimuli):
        best_matching_words = self.get_best_matching_words(stimuli)
        occurrences = [int(w == word) for w in best_matching_words]
        mean_occurrences = []
        for i in range(0, len(occurrences)):
            window = occurrences[max(0, i - 5):min(len(occurrences), i + 5)]
            mean_occurrences.append(int(sum(window) / len(window) > 0.5))
        return occurrences if self.language.stm == 'numeric' else mean_occurrences
        #alt = len([v for v, v_next in izip(occurrences, occurrences[1:]) if v != v_next])
        #alt_mean = len([v for v, v_next in izip(mean_occurrences, mean_occurrences[1:]) if v != v_next])

    def semantic_meaning(self, stimuli):
        return self.language.semantic_meaning(stimuli)

    def get_active_lexicon(self, stimuluses):
        best_matching_words = set(self.get_best_matching_words(stimuluses))
        active_lexicon = best_matching_words.difference({"?"})
        #logging.debug(active_lexicon)
        return active_lexicon

    def get_monotonicity(self, stimuluses):
        active_lexicon = self.get_active_lexicon(stimuluses)
        #logging.debug("Active lexicon of agent(%d): %s" % (self.id, active_lexicon))
        #mons = map(self.language.is_monotone, active_lexicon)
        mons = [self.language.is_monotone(word, stimuluses) for word in active_lexicon]
        #logging.debug("Monotonicity: %s" % mons)
        return mons.count(True)/len(mons) if len(mons) > 0 else 0.0

    def get_convexity(self, stimuluses):

        def check_convexity(word):
            meaning = self.pragmatic_meaning(word, stimuluses)
            alt = len([v for v, v_next in izip(meaning, meaning[1:]) if v != v_next])
            #logging.critical('Agent %d %s meaning (%d) = %s' % (self.id, word, alt <= 2, meaning))
            return alt <= 2

        best_matching_words = self.get_best_matching_words(stimuluses)
        #logging.critical("bmw: %s" % best_matching_words)
        active_words = set(best_matching_words).difference(set('?'))
        convexities = [check_convexity(w) for w in active_words]
        #logging.critical(convexities)
        convexity = convexities.count(True) / len(convexities) if len(convexities) > 0 else 0.0
        #logging.critical('Agent %d convexity: %d' % (self.id, convexity))
        return convexity

    def get_most_connected_word(self, category):
        return self.language.get_most_connected_word(category)

    def get_most_connected_category(self, word):
        return self.language.get_most_connected_category(word)

    def get_categories(self):
        return self.language.categories

    def get_best_matching_category(self, stimulus):
        return self.language.get_best_matching_category(stimulus)

    def get_categories_by_word(self, word):
        return self.language.get_categories_by_word(word)

    def get_lexicon(self):
        return self.language.lexicon

    def get_words_by_category(self, category):
        return self.language.get_words_by_category(category)

    def learn_stimulus(self, context, n):
        logging.debug("with discriminative success %f learns stimulus %d by " % (self.language.discriminative_success, n + 1))
        #logging.debug("category is not None: %d" % int(category is not None))
        #logging.debug("above threshold: %d" % int(self.language.discriminative_success >= self.language.discriminative_threshold))
        if self.language.discriminative_success >= self.language.discriminative_threshold:
            category = self.get_best_matching_category(context[n])
            logging.debug("updating category")
            self.language.update_category(category, context[n])
            #return category
        else:
            logging.debug("adding new category centered on %s" % context[n])
            self.language.add_category(context[n])

    def update_on_failure(self, word, category):
        self.language.decrement_word2category_connection(word, category)


class Speaker(Agent):

    def __init__(self, agent):
        Agent.__init__(self, agent.id, agent.language,
                       agent.cs1_scores, agent.cs2_scores, agent.cs12_scores)

    def update_on_success(self, word, category):
        self.language.increment_word2category_connection(word=word, category_index=category)
        self.language.inhibit_word2categories_connections(word=word, category_index=category)

    def update_on_success2c(self, word, category):
        logging.debug("Incrementing connections for %s, agent %d" % (word, self.id))
        csimilarities = [self.language.csimilarity(word, c) for c in self.language.categories]
        logging.debug("Speaker successful category %d, its similarity %f to %s meaning" % (category, csimilarities[category], word))
        logging.debug("Similarities: %s" % str(csimilarities))
        self.language.increment_word2category_connections_by_csimilarity(word, csimilarities)
        self.language.inhibit_word2categories_connections(word=word, category_index=category)

    def update_on_success_stage7(self, word, category):
        self.language.increment_word2category_connection(word=word, category_index=category)

    def add_new_word(self):
        return self.language.add_new_word()


class Hearer(Agent):
    def __init__(self, agent):
        Agent.__init__(self, agent.id, agent.language,
                       agent.cs1_scores, agent.cs2_scores, agent.cs12_scores)

    def get_topic(self, context, category):
        if category is None:
            raise ERROR

        category = self.language.categories[category]
        topic = category.select(context)
        #if topic is None:
        #    raise NO_DIFFERENCE_FOR_CATEGORY
        return topic

    # HEARER: The hearer computes the cardinalities ... of word forms ... defined as ... (STAGE 7)
    def select_word(self, category):
        threshold = .000  # todo
        words_by_category = self.language.get_words_sorted_by_val(category=category)

        if not len(words_by_category):
            return None, None

        word0 = words_by_category[0]
        categories0 = list(
            filter(lambda cat2propensity: cat2propensity[1] > threshold, self.language.get_categories_sorted_by_val(word0)))

        if len(words_by_category) == 1:
            return word0, categories0

        word1 = words_by_category[1]
        categories1 = list(
            filter(lambda cat2propensity: cat2propensity[1] > threshold, self.language.get_categories_sorted_by_val(word1)))

        logging.debug("Two words sorted by cardinality: %s, %s" % (word0, word1) if len(categories0) > len(categories1) else (word1, word0))
        return (word0, categories0) if len(categories0) > len(categories1) else (word1, categories1)

    def update_on_success(self, speaker_word, hearer_category):
        self.language.increment_word2category_connection(word=speaker_word, category_index=hearer_category)
        self.language.inhibit_category2words_connections(word=speaker_word, category_index=hearer_category)

    def update_on_success2c(self, word, category):
        logging.debug("Incrementing connections for %s, agent %d" % (word, self.id))
        csimilarities = [self.language.csimilarity(word, c) for c in self.language.categories]
        logging.debug("Hearer successful category %d, its similarity %f to %s meaning" % (self.get_categories()[category].id, csimilarities[category], word))
        logging.debug("c Similarities: %s" % str(csimilarities))
        self.language.increment_word2category_connections_by_csimilarity(word, csimilarities)
        self.language.inhibit_category2words_connections(word=word, category_index=category)

    def update_on_success_stage7(self, word, word_categories):
        for c_index, _ in word_categories:
            self.language.increment_word2category_connection(word, c_index)

    def add_word(self, word):
        return self.language.add_word(word)
