from objects.agent import Population
from objects.perception import Stimulus
from scripts.guessing_game import GuessingGame

params = {"populationSize": 2,
          "gamesPerRound": 1,  # gamesPerRound * 2 <= populationSize
          "learningRate": 0,  # co to?
          "discriminativeThreshold": 0.95,
          "weightDecay": 0.1,
          "steps": 50,
          "runs": 1}


class Simulation:

    def __init__(self, parameters=params):
        self.params = parameters

    def run(self):

        population = Population(self.params['populationSize'])

        for step in range(self.params["steps"]):
            print("STEP %d" % step)
            selected_pairs = population.select_pairs_per_round(self.params["gamesPerRound"])
            for speaker, hearer in selected_pairs:
                game = GuessingGame(speaker=speaker, hearer=hearer)
                game.play()

        for a in population.agents:
            print(len(a.categories))
            print(a.lexicon)
            print(a.lxc)


class RoundStatistics:
    discriminative_success = 0
    guessing_topic_success = 0
    guessing_word_success = 0


sim = Simulation()
sim.run()
