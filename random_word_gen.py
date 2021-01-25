import numpy.random as random


class RandomWordGen:

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

        with open(r'sorted_syllables.txt') as f:
            self.syllables = f.read().splitlines()
        self.gen_syllable = lambda: random.choice(self.syllables)

    def __call__(self):
        word_syllables_size = random.randint(2, 3)
        return ''.join(self.gen_syllable() for _ in range(word_syllables_size))


