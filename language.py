from __future__ import division  # force python 3 division in python 2
import logging
from guessing_game_exceptions import NO_WORD_FOR_CATEGORY, NO_SUCH_WORD, ERROR, NO_ASSOCIATED_CATEGORIES
from perception import Perception
from perception import Category
from perception import ReactiveUnit
from perception import Stimulus
from numpy import empty, array
from numpy import arange
from numpy import column_stack
from numpy import zeros
from numpy import row_stack
from numpy import linspace
from numpy import delete
from numpy import divide
from fractions import Fraction
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

# clone https://github.com/greghaskins/gibberish.git and run ~$ python setup.py install
from gibberish import Gibberish


class Language(Perception):
    gibberish = Gibberish()

    def __init__(self, params):
        Perception.__init__(self, params)
        self.lexicon = []
        self.lxc = AssociativeMatrix()
        self.delta_inc = params['delta_inc']
        self.delta_dec = params['delta_dec']
        self.delta_inh = params['delta_inh']

    def add_new_word(self):
        new_word = Language.gibberish.generate_word()
        self.add_word(new_word)
        return new_word

    def add_word(self, word):
        self.lexicon.append(word)
        self.lxc.add_row()

    def add_category(self, stimulus, weight=0.5):
        # print("adding discriminative category centered on %5.2f" % (stimulus.a/stimulus.b))
        c = Category(id = self.get_cat_id())
        c.add_reactive_unit(ReactiveUnit(stimulus), weight)
        self.categories.append(c)
        # TODO this should work
        self.lxc.add_col()
        return self.lxc.col_count() - 1  # this is the index of the added category

    def update_category(self, i, stimulus):
        logging.debug("updating category by adding reactive unit centered on %5.2f" % (stimulus.a / stimulus.b))
        self.categories[i].add_reactive_unit(ReactiveUnit(stimulus))

    def get_most_connected_word(self, category):
        if category is None:
            raise ERROR

        if not self.lexicon or all(v == 0 for v in self.lxc.get_row_by_col(category)):
            raise NO_WORD_FOR_CATEGORY
            # print("not words or all weights are zero")

        return self.get_words_sorted_by_val(category)[0]

    def get_words_sorted_by_val(self, category, threshold=-1):
        # https://stackoverflow.com/questions/1286167/is-the-order-of-results-coming-from-a-list-comprehension-guaranteed/1286180
        return [self.lexicon[index] for index, weigth in self.lxc.get_index2row_sorted_by_value(category) if
                weigth > threshold]

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
            c.weights = [w - self.alpha * w for w in c.weights]
        to_forget = [j for j in range(len(self.categories))
                     if max(self.categories[j].weights) < self.super_alpha and j != category_index]

        if len(to_forget):
            self.lxc.__matrix__ = delete(self.lxc.__matrix__, to_forget, axis=1)
            self.categories = list(delete(self.categories, to_forget))

    def forget_words(self, word_in_use):
        word_index = self.lexicon.index(word_in_use)
        to_forget = self.lxc.forget(self.alpha, self.super_alpha, word_index)
        self.lexicon = list(delete(self.lexicon, to_forget))

    def discrimination_game(self, context, topic):
        self.store_ds_result(Perception.Result.FAILURE)
        category_in_use = self.discriminate(context, topic)
        self.reinforce(category_in_use, context[topic])
        self.forget_categories(category_in_use)
        self.switch_ds_result()
        return self.categories.index(category_in_use)


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
        if max(self.__matrix__.flat) > 1.0:
            self.__matrix__ = divide(self.__matrix__, max(self.__matrix__.flat))

    def forget(self, forgetting_factor, super_alpha, word_index):
        self.__matrix__ = self.__matrix__ - self.__matrix__ * forgetting_factor

        to_forget = [j for j in range(self.__matrix__.shape[0])
                     if max(self.__matrix__[j]) < super_alpha and j != word_index]

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