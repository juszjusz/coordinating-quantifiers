from __future__ import division  # force python 3 division in python 2
import logging

from guessing_game_exceptions import NO_WORD_FOR_CATEGORY, NO_SUCH_WORD, ERROR, NO_ASSOCIATED_CATEGORIES
from perception import Perception
from perception import Category
from perception import ReactiveUnit
from perception import Stimulus
from numpy import empty
from numpy import arange
from numpy import column_stack
from numpy import zeros
from numpy import row_stack
from numpy import linspace
from numpy.random import choice
from numpy import array_equal
from fractions import Fraction
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

# clone https://github.com/greghaskins/gibberish.git and run ~$ python setup.py install
from gibberish import Gibberish


class Language(Perception):
    gibberish = Gibberish()

    def __init__(self):
        Perception.__init__(self)
        self.lexicon = []
        self.lxc = AssociativeMatrix()

    def add_new_word(self):
        new_word = Language.gibberish.generate_word()
        self.add_word(new_word)
        return new_word

    def add_word(self, word):
        self.lexicon.append(word)
        self.lxc.add_row()

    def add_category(self, stimulus, weight=0.5):
        # print("adding discriminative category centered on %5.2f" % (stimulus.a/stimulus.b))
        c = Category()
        c.add_reactive_unit(ReactiveUnit(stimulus), weight)
        self.categories.append(c)
        # TODO this should work
        self.lxc.add_col()
        return self.lxc.col_count() - 1  # this is the index of the added category

    def update_category(self, i, stimulus):
        # print("updating category by adding reactive unit centered on %5.2f" % (stimulus.a / stimulus.b))
        self.categories[i].add_reactive_unit(stimulus)

    def get_word(self, category):
        return self.get_words(category)[0]

    def get_words(self, category):
        if category is None:
            raise ERROR

        if not self.lexicon or all(v == 0 for v in self.lxc.get_row_by_col(category)):
            raise NO_WORD_FOR_CATEGORY
            # print("not words or all weights are zero")

        # TODO performance?
        # https://stackoverflow.com/questions/1286167/is-the-order-of-results-coming-from-a-list-comprehension-guaranteed/1286180
        return [self.lexicon[index] for index, _ in self.lxc.get_index2row_sorted_by_value(category)]

    def get_categories(self, word):
        word_index = self.lexicon.index(word)
        return self.lxc.get_index2col_sorted_by_value(word_index)

    def get_category(self, word):
        if word is None:
            raise ERROR

        if word not in self.lexicon:
            raise NO_SUCH_WORD

        # word_index = self.lexicon.index(word)
        index, max_propensity = self.get_categories(word)[0]

        # TODO still happens
        if max_propensity == 0:
            logging.debug("\"%s\" has no associated categories" % word)
            raise NO_ASSOCIATED_CATEGORIES

        return index

    def initialize_word2category_connection(self, word_index, category_index):
        self.lxc.set_value(word_index, category_index, .5)

    def increment_word2category_connection(self, word_index, category_index):
        value = self.lxc.get_value(word_index, category_index)
        self.lxc.set_value(word_index, category_index, value + .1 * value)

    def decrement_word2category_connection(self, word_index, category_index):
        value = self.lxc.get_value(word_index, category_index)
        self.lxc.set_value(word_index, category_index, value - .1 * value)

    # TODO deprecated
    def plot(self, filename=None, x_left=0, x_right=100, mode="Franek"):
        if not self.lxc.size():
            logging.debug("Language is empty")
            return
        if mode == 'Franek':
            forms_to_categories = {}
            for f in self.lexicon:
                forms_to_categories[f] = []
            for c in self.categories:
                j = self.categories.index(c)
                m = max(self.lxc.get_row_by_col(j))
                if m == 0:
                    continue
                else:
                    max_form_indices = [i for i, w in enumerate(self.lxc.get_row_by_col(j)) if w == m]
                    form = self.lexicon[max_form_indices[0]]
                    forms_to_categories[form].append(j)

            plt.title("language")
            plt.xscale("symlog")
            plt.yscale("symlog")
            ax = plt.gca()
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter())
            x = linspace(x_left, x_right, 20 * (x_right - x_left), False)
            colors = sns.color_palette()
            # sns.set_palette(colors)
            for i in range(len(self.lexicon)):
                f = self.lexicon[i]
                if len(forms_to_categories[f]) == 0:
                    continue
                else:
                    for j in forms_to_categories[f]:
                        ls = self.line_styles[i // len(colors)]
                        ci = i % len(colors)
                        plt.plot(x, [self.categories[j].fun(x_0) for x_0 in x],
                                 color=colors[ci], linestyle=ls)
                    plt.plot([], [], color=colors[ci], linestyle=ls, label=f)
            plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1, 1))
            plt.tight_layout(pad=0)
            plt.savefig(filename)
            plt.close()
        else:
            x = arange(x_left + 0.01, x_right, 0.01)
            logging.debug(x)
            f = [Fraction(int(100 * p), 100) for p in x]
            words = []
            n = 1
            for u in range(len(f)):
                fraction = f[u]
                s = Stimulus(fraction.numerator, fraction.denominator)
                responses = [c.response(s) for c in self.categories]
                m = max(responses)
                if m == 0:
                    words.append(0)
                else:
                    m_indices = [j for j, r in enumerate(responses) if r == m]
                    if len(m_indices) > 1:
                        logging.debug("more than one category responds with max")
                        logging.debug(len(m_indices))
                    m_i = m_indices[0]
                    w, e = self.get_word(m_i)
                    words.append(e if w is None else self.lexicon.index(w) + 1)
            plt.xlabel("ratio")
            plt.ylabel("word")
            plt.ylim(bottom=Language.Error._END_)
            plt.ylim(top=len(self.lexicon))
            plt.xlim(left=x_left)
            plt.xlim(right=x_right)
            # locs, labels = plt.yticks()
            plt.plot(x, words, 'o')
            plt.legend(['data'], loc='best')
            if filename is None:
                plt.show()
            else:
                plt.savefig(filename)
            plt.close()


class AssociativeMatrix():
    def __init__(self):
        self.matrix = empty(shape=(0, 0))

    def add_row(self):
        self.matrix = row_stack((self.matrix, zeros(self.col_count())))

    def add_col(self):
        self.matrix = column_stack((self.matrix, zeros(self.row_count())))

    def get_row_by_col(self, column):
        return self.matrix[0::, column]

    # returns row vector with indices sorted by values in reverse order, i.e. [(index5, 1000), (index100, 999), (index500,10), ...]
    def get_index2row_sorted_by_value(self, column):
        indices = range(self.row_count())
        row = self.get_row_by_col(column)
        return sorted(zip(indices, row), key=lambda index2row: index2row[1], reverse=True)

    def get_col_by_row(self, row):
        return self.matrix[row, 0::]

    # returns col vector with indices sorted by values in reverse order, i.e. [(index5, 1000), (index100, 999), (index500,10), ...]
    def get_index2col_sorted_by_value(self, row):
        indices = range(self.col_count())
        col = self.get_col_by_row(row)
        return sorted(zip(indices, col), key=lambda index2col: index2col[1], reverse=True)

    def col_count(self):
        return self.matrix.shape[1]

    def row_count(self):
        return self.matrix.shape[0]

    def get_value(self, row, col):
        return self.matrix[row][col]

    def set_value(self, row, col, value):
        self.matrix[row][col] = value

    def size(self):
        return self.size()
