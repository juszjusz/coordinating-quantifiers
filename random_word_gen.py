import numpy.random as random
import os

from pathlib import Path
from path_provider import PathProvider


class RandomWordGen:

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

        path = Path(os.path.abspath('.')).joinpath('sorted_syllables.txt')
        with open(path) as f:
            self.syllables = f.read().splitlines()
        self.gen_syllable = lambda: random.choice(self.syllables)

    def __call__(self):
        word_syllables_size = random.randint(2, 3)
        return ''.join(self.gen_syllable() for _ in range(word_syllables_size))


