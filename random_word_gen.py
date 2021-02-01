import pathlib

import numpy.random as random


class RandomWordGen:

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

        root_path = pathlib.Path(__file__).parent.absolute()
        path = root_path.joinpath('sorted_syllables.txt')

        with open(path) as f:
            self.syllables = f.read().splitlines()
        self.gen_syllable = lambda: random.choice(self.syllables)

    def __call__(self):
        word_syllables_size = random.randint(2, 3)
        return ''.join(self.gen_syllable() for _ in range(word_syllables_size))


