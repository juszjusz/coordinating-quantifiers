from objects.perception import DiscriminativeCategory
from objects.perception import ReactiveUnit
from objects.perception import Stimulus


class Language:

    def __init__(self, lexicon=[], discriminative_categories=[], mapping=None):
        self.lexicon = lexicon
        self.discriminative_categories = discriminative_categories
        self.mapping = mapping

    def add_discriminative_category(self, stimulus: Stimulus, weight=0.5):
        # TODO modify mapping
        # print("adding discriminative category centered on %5.2f" % (stimulus.a/stimulus.b))
        c = DiscriminativeCategory()
        r = ReactiveUnit(stimulus)
        c.add_reactive_unit(r, weight)
        self.discriminative_categories.append(c)

    def pick_word(self, category):
        # TODO
        return None

    def pick_category(self, word):
        # TODO
        return None
