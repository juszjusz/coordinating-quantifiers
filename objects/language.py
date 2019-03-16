import logging
import random

class Language():
    def __init__(self, word2categories: {str: list}):
        self.word2categories = word2categories
        self.category2words = self.__inverse_dict(self.word2categories)
        self.word2category_weigth = self.__intitialize_word2category_weigths(self.word2categories)

    #pull method to some kind of python interface
    def pick_word(self, category):
        max = {'word': None, 'score': -1}
        for word in self.category2words[category]:
            # TODO just proof of concept
            score = self.word2category_weigth[word + '#' + category]
            if score > max['score']:
                max['word'] = word
                max['score'] = score

        logging.log(logging.INFO, 'naming category')
        return max['word']

    def pick_category(self, word):
        max = {'category': None, 'score': -1}
        for category in self.word2categories[word]:
            score = self.word2category_weigth[word + '#' + category]
            if score > max['score']:
                max['category'] = category
                max['score'] = score

        return max['category']

    @staticmethod
    def __inverse_dict(word2categories: {str: list}):
        distinct_categories = set(item for sublist in word2categories.values() for item in sublist)
        category2words = {category: [] for category in distinct_categories}
        for category, words in category2words.items():
            for word, categories in word2categories.items():
                if category in categories:
                    words.append(word)
        return category2words

    @staticmethod
    def __intitialize_word2category_weigths(word2categories: {str: list}):
        word2category_weigth = {}
        for word, categories in word2categories.items():
            for category in categories:
                # TODO just proof of concept
                word2category_weigth[word + '#' + category] = random.randint(0, 100)
        return word2category_weigth

    def pick_random_category(self):
        return random.choice(list(self.category2words))

    def contains_word(self, word):
        return word in self.word2categories