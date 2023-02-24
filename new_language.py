from __future__ import division  # force python 3 division in python 2
import logging
from guessing_game_exceptions import NO_WORD_FOR_CATEGORY, NO_SUCH_WORD, ERROR, NO_ASSOCIATED_CATEGORIES
from perception import Perception
from perception import Category
from numpy import empty, array, minimum
from numpy import column_stack
from numpy import zeros
from numpy import row_stack
from numpy import delete
from numpy import divide
from itertools import izip


# clone https://github.com/greghaskins/gibberish.git and run ~$ python setup.py install


class NewLanguage:
    def __init__(self, params):

    # self.stm = params['stimulus']
    # self.delta_inc = params['delta_inc']
    # self.delta_dec = params['delta_dec']
    # self.delta_inh = params['delta_inh']
    # self.discriminative_threshold = params['discriminative_threshold']
    # self.alpha = params['alpha']  # forgetting
    # self.beta = params['beta']  # learning rate
    # self.super_alpha = params['super_alpha']

    def add_word(self, word):
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
        # self.__matrix__ = self.__matrix__ - self.__matrix__ * forgetting_factor

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
