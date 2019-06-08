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
            selected_pairs = population.select_pairs_per_round(self.parameters["gamesPerRound"])
            for speaker, hearer in selected_pairs:
                GuessingGame().play_round(speaker, hearer, None)


class RoundStatistics():
    discriminative_success = 0
    guessing_topic_success = 0
    guessing_word_success = 0