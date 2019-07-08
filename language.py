from perception import Category
from perception import ReactiveUnit
from perception import Stimulus
from numpy import empty
from numpy.random import choice
# from enum import Enum

# clone https://github.com/greghaskins/gibberish.git and run ~$ python setup.py install
from gibberish import Gibberish


class Language:

    class Error:
        NO_ERROR = 0
        NO_CATEGORY = 1            # agent has no categories
        NO_DISCRIMINATION = 2      # agent has categories but is unable to discriminate
        NO_WORD_FOR_CATEGORY = 3   # agent has no word for category
        NO_SUCH_WORD = 4           # agent doesn't know the word
        NO_DIFFERENCE = 5          # agent fails to select topic using category bcs it produces the same responses for both stimuli
        ERROR = 7

    gibberish = Gibberish()

    def __init__(self):
        self.lexicon = []
        self.categories = []
        self.lxc = empty(shape=(0, 0))

    def add_new_word(self):
        new_word = Language.gibberish.generate_word()
        self.add_word(new_word)
        return new_word
        #new_word_index = rows_cnt
        #self.lxc[new_word_index, category] = 0.5

    def add_word(self, word):
        self.lexicon.append(word)
        rows_cnt, cols_cnt = self.lxc.shape
        self.lxc.resize((rows_cnt + 1, cols_cnt), refcheck=False)

    def add_category(self, stimulus, weight=0.5):
        # print("adding discriminative category centered on %5.2f" % (stimulus.a/stimulus.b))
        c = Category()
        c.add_reactive_unit(ReactiveUnit(stimulus), weight)
        self.categories.append(c)
        rows_cnt, cols_cnt = self.lxc.shape
        self.lxc.resize((rows_cnt, cols_cnt+1), refcheck=False)

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
        max_propensity_indices = [i for i, j in enumerate(propensities) if j == max_propensity]
        # TODO random choice?
        return choice(max_propensity_indices), Language.Error.NO_ERROR
