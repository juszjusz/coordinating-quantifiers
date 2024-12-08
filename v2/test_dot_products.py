import unittest
from fractions import Fraction

from calculator import QuotientCalculator


class TestDotProducts(unittest.TestCase):
    def test(self):
        stimuli0 = Fraction(8, 37)
        stimuli_1058 = Fraction(8, 23)
        stimuli_1059 = Fraction(31, 89)
        stimuli_1060 = Fraction(23, 66)

        stimuli, density, calculator = QuotientCalculator.from_description_with_ans()

        monotonic = [calculator.dot_product(stimuli0, stimuli_1058),
                     calculator.dot_product(stimuli0, stimuli_1059),
                     calculator.dot_product(stimuli0, stimuli_1060)]

        print(monotonic)
        # Should be monotonic, but it is not!!!
        self.assertLess(calculator.dot_product(stimuli0, stimuli_1058), calculator.dot_product(stimuli0, stimuli_1059))
        self.assertLess(calculator.dot_product(stimuli0, stimuli_1059), calculator.dot_product(stimuli0, stimuli_1060))
