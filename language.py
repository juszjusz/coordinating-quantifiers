from __future__ import division  # force python 3 division in python 2
import logging

from picklable_itertools import izip

from guessing_game_exceptions import NO_WORD_FOR_CATEGORY, NO_SUCH_WORD, ERROR, NO_ASSOCIATED_CATEGORIES
from perception import Perception
from perception import Category
from numpy import empty, array, minimum
from numpy import column_stack
from numpy import zeros
from numpy import row_stack
from numpy import delete
from numpy import divide

class Language(Perception):

    def __init__(self, params, word_gen):
        Perception.__init__(self)
        self.lexicon = []
        self.word_gen = word_gen
        self.lxc = AssociativeMatrix()
        self.stm = params['stimulus']
        self.delta_inc = params['delta_inc']
        self.delta_dec = params['delta_dec']
        self.delta_inh = params['delta_inh']
        self.discriminative_threshold = params['discriminative_threshold']
        self.alpha = params['alpha']  # forgetting
        self.beta = params['beta']  # learning rate
        self.super_alpha = params['super_alpha']

    def add_new_word(self):
        new_word = self.word_gen()
        self.add_word(new_word)
        return new_word

    def add_word(self, word):
        self.lexicon.append(word)
        self.lxc.add_row()

    def add_category(self, stimulus, weight=0.5):
        # print("adding discriminative category centered on %5.2f" % (stimulus.a/stimulus.b))
        c = Category(id=self.get_cat_id())
        c.add_reactive_unit(stimulus, weight)
        self.categories.append(c)
        # TODO this should work
        self.lxc.add_col()
        return self.lxc.col_count() - 1  # this is the index of the added category

    def update_category(self, i, stimulus):
        logging.debug("updating category by adding reactive unit centered on %s" % stimulus)
        self.categories[i].add_reactive_unit(stimulus)

    def get_most_connected_word(self, category):
        if category is None:
            raise ERROR

        if not self.lexicon or all(v == 0.0 for v in self.lxc.get_row_by_col(category)):
            raise NO_WORD_FOR_CATEGORY
            # print("not words or all weights are zero")

        return self.get_words_sorted_by_val(category)[0]

    def get_words_sorted_by_val(self, category, threshold=-1):
        # https://stackoverflow.com/questions/1286167/is-the-order-of-results-coming-from-a-list-comprehension-guaranteed/1286180
        return [self.lexicon[index] for index, weight in self.lxc.get_index2row_sorted_by_value(category) if
                weight > threshold]

    def get_categories_sorted_by_val(self, word):
        word_index = self.lexicon.index(word)
        return self.lxc.get_index2col_sorted_by_value(word_index)

    def get_categories_by_word(self, word):
        word_index = self.lexicon.index(word)
        return self.lxc.get_col_by_row(word_index)

    def get_words_by_category(self, category):
        return self.lxc.get_row_by_col(category)

    def get_most_connected_category(self, word):
        if word is None:
            raise ERROR

        if word not in self.lexicon:
            raise NO_SUCH_WORD

        category_index, max_propensity = self.get_categories_sorted_by_val(word)[0]

        # TODO still happens
        if max_propensity == 0:
            logging.debug("\"%s\" has no associated categories" % word)
            raise NO_ASSOCIATED_CATEGORIES

        return category_index

    def initialize_word2category_connection(self, word, category_index):
        word_index = self.lexicon.index(word)
        self.lxc.set_value(word_index, category_index, .5)

    def increment_word2category_connection(self, word, category_index):
        word_index = self.lexicon.index(word)
        value = self.lxc.get_value(word_index, category_index)
        self.lxc.set_value(word_index, category_index, value + self.delta_inc * value)

    def inhibit_word2category_connection(self, word, category_index):
        word_index = self.lexicon.index(word)
        value = self.lxc.get_value(word_index, category_index)
        self.lxc.set_value(word_index, category_index, value - self.delta_inh * value)

    def inhibit_word2categories_connections(self, word, category_index):
        for k_index, _ in self.get_categories_sorted_by_val(word):
            if k_index != category_index:
                self.inhibit_word2category_connection(word, k_index)

    def inhibit_category2words_connections(self, word, category_index):
        for v in self.get_words_sorted_by_val(category_index):
            if v != word:
                self.inhibit_word2category_connection(word=v, category_index=category_index)

    def decrement_word2category_connection(self, word, category_index):
        word_index = self.lexicon.index(word)
        value = self.lxc.get_value(word_index, category_index)
        self.lxc.set_value(word_index, category_index, value - self.delta_dec * value)

    def forget_categories(self, category_in_use):
        category_index = self.categories.index(category_in_use)
        for c in self.categories:
            c.decrement_weights(self.alpha)
        to_forget = [j for j in range(len(self.categories))
                     if self.categories[j].max_weigth() < self.super_alpha and j != category_index]

        if len(to_forget):
            self.lxc.__matrix__ = delete(self.lxc.__matrix__, to_forget, axis=1)
            self.categories = list(delete(self.categories, to_forget))

    def forget_words(self):
        to_forget = self.lxc.forget(0.01)
        self.lexicon = list(delete(self.lexicon, to_forget))

    def discrimination_game(self, context, topic):
        self.store_ds_result(False)
        winning_category = self.discriminate(context, topic)
        winning_category.reinforce(context[topic], self.beta)
        self.forget_categories(winning_category)
        self.switch_ds_result()
        return self.categories.index(winning_category)

    def increment_word2category_connections_by_csimilarity(self, word, csimilarities):
        row = self.lexicon.index(word)
        increments = [sim * self.delta_inc * (sim > 0.25) for sim in csimilarities]
        #logging.debug("Increments: %s" % str(increments))

        old_weights = self.lxc.get_col_by_row(self.lexicon.index(word))
        #logging.debug("Old weights: %s" % str(old_weights))

        incremented_weights = [weight + inc for weight, inc in zip(old_weights, increments)]
        #logging.debug("Incremented weights: %s" % str(incremented_weights))
        self.lxc.set_values(axis=0, index=row, values=incremented_weights)

    # based on how much the word meaning covers the category
    def csimilarity(self, word, category):
        area = category.union()
        # omit multiplication by x_delta because all we need is ratio: coverage/area:
        word_meaning = self.word_meaning(word)
        coverage = minimum(word_meaning, area)

        return sum(coverage) / sum(area)

    def word_meaning(self, word):
        word_index = self.lexicon.index(word)
        return sum([category.union() * word2category_weigth for category, word2category_weigth in zip(self.categories, self.lxc.__matrix__[word_index])])

    def semantic_meaning(self, word, stimuli):
        word_index = self.lexicon.index(word)
        activations = [sum([float(c.response(s) > 0.0) * float(self.lxc.get_value(word_index, self.categories.index(c)) > 0.0) for c in self.categories]) for s in stimuli]
        flat_bool_activations = map(lambda x: int(x > 0.0), activations)
        mean_bool_activations = []
        for i in range(0, len(flat_bool_activations)):
            window = flat_bool_activations[max(0, i - 5):min(len(flat_bool_activations), i + 5)]
            mean_bool_activations.append(int(sum(window)/len(window) > 0.5))
        #logging.critical("Word %s ba: %s" % (word, bool_activations))
        #logging.critical("Word %s mba: %s" % (word, mean_bool_activations))
        return mean_bool_activations if self.stm == 'quotient' else flat_bool_activations
        #return flat_bool_activations

    def is_monotone(self, word, stimuli):
        bool_activations = self.semantic_meaning(word, stimuli)
        alt = len([a for a, aa in izip(bool_activations, bool_activations[1:]) if a != aa])
        return alt == 1


class AssociativeMatrix:
    def __init__(self, initial_size=(0, 0)):
        self.__matrix__ = empty(shape=initial_size)
        self.__max_shape__ = initial_size

    def add_row(self):
        self.__matrix__ = row_stack((self.__matrix__, zeros(self.col_count())))
        self.__max_shape__ = (max(self.__matrix__.shape[0], self.__max_shape__[0]), self.__max_shape__[1])

    def add_col(self):
        self.__matrix__ = column_stack((self.__matrix__, zeros(self.row_count())))
        self.__max_shape__ = (self.__max_shape__[0], max(self.__matrix__.shape[1], self.__max_shape__[1]))

    def get_row_by_col(self, column):
        return self.__matrix__[0::, column]

    # returns row vector with indices sorted by values in reverse order, i.e. [(index5, 1000), (index100, 999), (index500,10), ...]
    def get_index2row_sorted_by_value(self, column):
        index2rows = enumerate(self.get_row_by_col(column))
        return sorted(index2rows, key=lambda index2row: index2row[1], reverse=True)

    def get_col_by_row(self, row):
        return self.__matrix__[row, 0::]

    # returns col vector with indices sorted by values in reverse order, i.e. [(index5, 1000), (index100, 999), (index500,10), ...]
    def get_index2col_sorted_by_value(self, row):
        index2cols = enumerate(self.get_col_by_row(row))
        return sorted(index2cols, key=lambda index2col: index2col[1], reverse=True)

    def col_count(self):
        return self.__matrix__.shape[1]

    def row_count(self):
        return self.__matrix__.shape[0]

    def get_value(self, row, col):
        return self.__matrix__[row][col]

    def set_value(self, row, col, value):
        self.__matrix__[row][col] = value

    def set_values(self, axis, index, values):
        if axis == 0:
            self.__matrix__[index] = values
        else:
            self.__matrix__[:, index] = values

    def normalize(self, axis, index):
        if axis == 0:
            if max(self.__matrix__[index].flat) > 1.0:
                self.__matrix__[index] = divide(self.__matrix__[index], max(self.__matrix__[index].flat))
        else:
            if max(self.__matrix__[:, index].flat) > 1.0:
                self.__matrix__[:, index] = divide(self.__matrix__[:, index], max(self.__matrix__[:, index].flat))

    def forget(self, super_alpha):
        #self.__matrix__ = self.__matrix__ - self.__matrix__ * forgetting_factor

        to_forget = [j for j in range(self.__matrix__.shape[0]) if max(self.__matrix__[j]) < super_alpha]

        if len(to_forget):
            self.__matrix__ = delete(self.__matrix__, to_forget, axis=0)

        return to_forget

    def size(self):
        return self.__matrix__.size

    def max_shape(self):
        return self.__max_shape__

    def delete_col(self, col):
        self.__matrix__ = delete(self.__matrix__, col, axis=1)

    def delete_row(self, row):
        self.__matrix__ = delete(self.__matrix__, row, axis=0)

    def to_array(self):
        return array(self.__matrix__)

    def to_matrix(self):
        return self.__matrix__