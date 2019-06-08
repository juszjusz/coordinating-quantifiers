import random
import objects.agent
from scripts.game_definitions import GuessingGame
from objects.agent import Population

class Simualtion:
    step = 0
    parameters = { "populationSize": 10,
                   "gamesPerRound": 2, #gamesPerRound * 2 <= populationSize
                   "learningRate": 0, #co to?
                   "discriminativeThreshold": 0.95,
                   "weightDecay": 0.1,
                   "steps": 5}
    statistics = []

    def run(self):
        population = Population(self.parameters["populationSize"])

        for step in range(self.parameters["steps"]):
            population.select_pairs()
            for j in range(self.parameters["gamesPerRound"]):
                speaker, hearer = population.get_speaker_and_hearer()#czy tutaj podawac nr gry?
                GuessingGame().play_round(speaker, hearer, None)


class RoundStatistics():
    discriminative_success = 0
    guessing_topic_success = 0
    guessing_word_success = 0