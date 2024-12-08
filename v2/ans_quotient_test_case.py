from fractions import Fraction
from itertools import groupby
import numpy as np
from calculator import QuotientCalculator
from domain_objects import NewCategory

if __name__ == '__main__':
    fs = list(set([Fraction(n, k) for k in range(1, 101) for n in range(1, k)]))
    fs = sorted(fs)
    fs_indices = {f: i for i, f in enumerate(fs)}
    fs_indices_reverse = {i: f for i, f in enumerate(fs)}

    support, density, calculator = QuotientCalculator.from_description_with_ans()
    # category1 = NewCategory.init_from_stimuli([(39460.72005664417, Fraction(2, 49))])
    # category2 = NewCategory.init_from_stimuli([(10573.132438794533, Fraction(45, 98))])
    # category4 = NewCategory.init_from_stimuli([(37542.85367026418, Fraction(8, 37))])

    category2 = NewCategory.init_from_stimuli([(1057, Fraction(45, 98))])
    category4 = NewCategory.init_from_stimuli([(3754, Fraction(8, 37))])

    categories = [category2, category4]
    responses = [category.response_all(calculator) for category in categories]
    stimuli_response_maximizers = np.argmax(responses, axis=0)

    category2stimuli = [(category, stimuli) for stimuli, category in enumerate(stimuli_response_maximizers)]

    category2stimuli = sorted(category2stimuli, key=lambda c2s: c2s[0])
    category2stimuli = {category: [s for c, s in v] for category, v in
                        groupby(category2stimuli, key=lambda c2s: c2s[0])}

    stimuli_1058 = Fraction(8, 23)
    stimuli_1059 = Fraction(31, 89)
    stimuli_1060 = Fraction(23, 66)

    # Here is a problem with a monotonicity, reaction cat to stimuli might be non monotnic, but when category consists
    # of  more than 1 stimuli, which is not a case in this example.
    # I think the problem might be a numeric due to unrestricted weight?
    c2_s1058_response = category2.response(stimuli_1058, calculator)
    c4_s1058_response = category4.response(stimuli_1058, calculator)
    calculator.dot_product()

    c2_s1059_response = category2.response(stimuli_1059, calculator)
    c4_s1059_response = category4.response(stimuli_1059, calculator)

    c2_s1060_response = category2.response(stimuli_1060, calculator)
    c4_s1060_response = category4.response(stimuli_1060, calculator)
    # stimuli_response_maximizers = set(stimuli_response_maximizers)
    print()
    # QuotientCalculator.from_description_with_ans()
