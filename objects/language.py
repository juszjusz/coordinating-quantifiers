from objects.perception import Category
from objects.perception import ReactiveUnit
from objects.perception import Stimulus
from numpy import array
from numpy.random import choice

# clone https://github.com/greghaskins/gibberish.git and run ~$ python setup.py install
from gibberish import Gibberish


class Language:

    gibberish = Gibberish()

    def __init__(self, lexicon=[], categories=[], lxc=array([])):
        self.lexicon = lexicon
        self.categories = categories
        self.lxc = lxc

    def add_category(self, stimulus: Stimulus, weight=0.5):
        # print("adding discriminative category centered on %5.2f" % (stimulus.a/stimulus.b))
        c = Category()
        c.add_reactive_unit(ReactiveUnit(stimulus), weight)
        self.categories.append(c)
        rows_cnt = self.lxc.shape[0] if self.lxc.ndim == 2 else 0 if self.lxc.shape[0] == 0 else 1
        cols_cnt = self.lxc.shape[1] if self.lxc.ndim == 2 else self.lxc.shape[0]
        # print("rows = %d, cols = %d" % (rows_cnt, cols_cnt))
        self.lxc.resize((rows_cnt, cols_cnt+1), refcheck=False)

    def update_category(self, i: int, stimulus: Stimulus):
        # print("updating category by adding reactive unit centered on %5.2f" % (stimulus.a / stimulus.b))
        self.language.categories[i].add_reactive_unit(stimulus)

    def get_word(self, category):
        # print("get_word")
        if not self.lexicon or all(v == 0 for v in self.lxc[0::, category]):
            # print("not words or all weights are zero")
            new_word = Language.gibberish.generate_word()
            self.lexicon.append(new_word)
            rows_cnt, cols_cnt = self.lxc.shape
            self.lxc.resize((rows_cnt+1, cols_cnt), refcheck=False)
            new_word_index = rows_cnt
            self.lxc[new_word_index, category] = 0.5
            return None

        # TODO performance?
        word_propensities = self.lxc[0::, category]
        max_propensity = max(word_propensities)
        max_propensity_indices = [i for i, j in enumerate(word_propensities) if j == max_propensity]
        return self.lexicon[choice(max_propensity_indices)]

    def get_category(self, word):
        # TODO
        return None