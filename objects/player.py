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

    def pick_word(self, category):
        return self.language.pick_word(category)

    def pick_category(self, context: (int, int), topic):
        #TODO
        return None


class HearerRole(Player):
    def __init__(self, id, language: Language):
        Player.__init__(self, id, language)

    def pick_category(self, word):
        return self.language.pick_category(word)

    def pick_stimulus(self, category):
        #TODO
        return None


class PlayersPair():
    def __init__(self, player1: Player, player2: Player):
        self.__players = (player1, player2)

    def pick_speaker_and_hearer(self):
        zero_or_one = random.randint(0, 1)
        print('picked objects', self.__players[zero_or_one].id, 'as speaker')
        print('picked objects', self.__players[1 - zero_or_one].id, 'as hearer')
        return self.__players[zero_or_one]._as_speaker(), self.__players[1 - zero_or_one]._as_hearer()
