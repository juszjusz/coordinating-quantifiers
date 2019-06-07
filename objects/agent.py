from objects.language import Language
import random

class Population:

    agents = []
    pair = 0

    def __init__(self, population_size: int):
        self.population = []
        for i in range(population_size):
            self.population.append(Agent(i))


    def select_pairs(self):
        self.agents = random.sample(range(self.parameters["populationSize"]), self.parameters["gamesPerRound"] * 2)
        pair = 0

    def get_speaker_and_hearer(self):
        speaker_and_hearer = self.agents[2*self.pair]._as_speaker(), self.agents[2*self.pair+1]._as_hearer()
        self.pair = self.pair + 1
        return speaker_and_hearer


class Agent:
    def __init__(self, id, language: Language = Language()):
        self.language = language
        self.id = id

    def _as_speaker(self):
        return SpeakerAgent(self.id, self.language)

    def _as_hearer(self):
        return HearerAgent(self.id, self.language)


class SpeakerAgent(Agent):
    def __init__(self, id, language: Language):
        Agent.__init__(self, id, language)

    def get_word(self, category):
        return self.language.pick_word(category)

    # getDiscriminativeCategory
    def get_discriminative_category(self, context: (int, int), topic):
        #TODO
        return None


class HearerAgent(Agent):
    def __init__(self, id, language: Language):
        Agent.__init__(self, id, language)

    def get_discriminative_category(self, word):
        return self.language.pick_category(word)

    def get_stimulus(self, category):
        #TODO
        return None


class PlayersPair():
    def __init__(self, player1: Agent, player2: Agent):
        self.__players = (player1, player2)

    def get_speaker_and_hearer(self):
        zero_or_one = random.randint(0, 1)
        print('picked objects', self.__players[zero_or_one].id, 'as speaker')
        print('picked objects', self.__players[1 - zero_or_one].id, 'as hearer')
        return self.__players[zero_or_one]._as_speaker(), self.__players[1 - zero_or_one]._as_hearer()
