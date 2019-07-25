class LanguageError(Exception):
    pass


class PerceptionError(Exception):
    pass


class ERROR(LanguageError):
    pass


# NO_WORD_FOR_CATEGORY = Perception.Error._END_ - 1  # agent has no word for category
class NO_WORD_FOR_CATEGORY(LanguageError):
    pass


# NO_SUCH_WORD = Perception.Error._END_ - 2  # agent doesn't know the word
class NO_SUCH_WORD(LanguageError):
    pass


# agent has not associated categories with a word
class NO_ASSOCIATED_CATEGORIES(LanguageError):
    pass


# agent has no categories
class NO_CATEGORY(PerceptionError):
    pass


# agent has categories but is unable to discriminate, lower response for stimulus 1
class NO_DISCRIMINATION_LOWER_1(PerceptionError):
    pass


# agent has categories but is unable to discriminate, lower response for stimulus 2
class NO_DISCRIMINATION_LOWER_2(PerceptionError):
    pass


# agent fails to select topic using category bcs it produces the same responses for both stimuli
class NO_DIFFERENCE_FOR_CATEGORY(PerceptionError):
    pass


# agent has categories but they return 0 as response for stimulus 1
class NO_POSITIVE_RESPONSE_1(PerceptionError):
    pass


# agent has categories but they return 0 as response for stimulus 2
class NO_POSITIVE_RESPONSE_2(PerceptionError):
    pass


# stimuli are indistinguishable for agent perception (jnd)
class NO_NOTICEABLE_DIFFERENCE(PerceptionError):
    pass
