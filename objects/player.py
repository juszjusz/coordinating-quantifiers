from objects.language import Language
import random


class Player():
    def __init__(self, id, language: Language):
        self.language = language
        self.id = id

    def _as_speaker(self):
        return SpeakerRole(self.id, self.language)

    def _as_hearer(self):
        return HearerRole(self.id, self.language)


class SpeakerRole(Player):
    def __init__(self, id, language: Language):
        Player.__init__(self, id, language)

    # def pick_discriminating_word(self, context: (int, int), topic: int):
    #     print('playing discrimination scripts with contexts')
    #     category = self.pick_category(context, topic)
    #     word = self.language.pick_word(category)
    #     return word

    def pick_word(self, category):
        return self.language.pick_word(category)

    def pick_category(self, context: (int, int), topic):
        print('playing discrimination scripts that results with discriminating context for stimuli1, stimuli2')
        return self.language.pick_random_category()


class HearerRole(Player):
    def __init__(self, id, language: Language):
        Player.__init__(self, id, language)

    def pick_category(self, word):
        if self.language.contains_word(word):
            print('i have given word', word, 'in my dictionary')
        else:
            print('dont have', word, 'in my dictionary')
        return self.language.pick_category(word)

    def pick_stimulus(self, category):
        return -1


class PlayersPair():
    def __init__(self, player1: Player, player2: Player):
        self.__players = (player1, player2)

    def pick_speaker_and_hearer(self):
        zero_or_one = random.randint(0, 1)
        print('picked objects', self.__players[zero_or_one].id, 'as speaker')
        print('picked objects', self.__players[1 - zero_or_one].id, 'as hearer')
        return self.__players[zero_or_one]._as_speaker(), self.__players[1 - zero_or_one]._as_hearer()
