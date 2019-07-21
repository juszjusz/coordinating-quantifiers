from __future__ import division  # force python 3 division in python 2
import logging
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
from fractions import Fraction
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

# clone https://github.com/greghaskins/gibberish.git and run ~$ python setup.py install
from gibberish import Gibberish


class Language(Perception):
    class Error(Perception.Error):
        NO_WORD_FOR_CATEGORY = Perception.Error._END_ - 1  # agent has no word for category
        NO_SUCH_WORD = Perception.Error._END_ - 2  # agent doesn't know the word
        _END_ = NO_SUCH_WORD

    gibberish = Gibberish()

    def __init__(self):
        Perception.__init__(self)
        self.lexicon = []
        self.lxc = empty(shape=(0, 0))

    def add_new_word(self):
        new_word = Language.gibberish.generate_word()
        self.add_word(new_word)
        return new_word

    def add_word(self, word):
        self.lexicon.append(word)
        self.lxc = row_stack((self.lxc, zeros(self.lxc.shape[1])))

    def add_category(self, stimulus, weight=0.5):
        # print("adding discriminative category centered on %5.2f" % (stimulus.a/stimulus.b))
        c = Category()
        c.add_reactive_unit(ReactiveUnit(stimulus), weight)
        self.categories.append(c)
        # TODO this should work
        self.lxc = column_stack((self.lxc, zeros(self.lxc.shape[0])))
        return self.lxc.shape[1] - 1  # this is the index of the added category

    def update_category(self, i, stimulus):
        # print("updating category by adding reactive unit centered on %5.2f" % (stimulus.a / stimulus.b))
        self.categories[i].add_reactive_unit(stimulus)

    def get_word(self, category):
        if category is None:
            return None, Language.Error.ERROR

        if not self.lexicon or all(v == 0 for v in self.lxc[0::, category]):
            return None, Language.Error.NO_WORD_FOR_CATEGORY
            # print("not words or all weights are zero")

        # TODO performance?
        word_propensities = self.lxc[0::, category]
        max_propensity = max(word_propensities)
        max_propensity_indices = [i for i, j in enumerate(word_propensities) if j == max_propensity]
        return self.lexicon[choice(max_propensity_indices)], Language.Error.NO_ERROR

    def get_category(self, word):
        if word is None:
            return None, Language.Error.ERROR

        if word not in self.lexicon:
            return None, Language.Error.NO_SUCH_WORD
        word_index = self.lexicon.index(word)
        propensities = self.lxc[word_index, 0::]
        max_propensity = max(propensities)

        # TODO still happens
        if max_propensity == 0:
            logging.debug("\"%s\" has no associated categories" % word)
            logging.debug("category propensities: ")
            logging.debug(propensities)
            raise Exception("no associated categories")

        max_propensity_indices = [i for i, j in enumerate(propensities) if j == max_propensity]
        # TODO random choice?
        return choice(max_propensity_indices), Language.Error.NO_ERROR

    def plot(self, filename=None, x_left=0, x_right=100, mode="Franek"):
        if not self.lxc.size:
            logging.debug("Language is empty")
            return
        if mode == 'Franek':
            forms_to_categories = {}
            for f in self.lexicon:
                forms_to_categories[f] = []
            for c in self.categories:
                j = self.categories.index(c)
                m = max(self.lxc[0::, j])
                if m == 0:
                    continue
                else:
                    max_form_indices = [i for i, w in enumerate(self.lxc[0::, j]) if w == m]
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