from __future__ import division # force python 3 division in python 2
from agent import Population
from guessing_game import GuessingGame
import cProfile
import matplotlib.pyplot as plt

params = {"populationSize": 2,
          "gamesPerRound": 1,  # gamesPerRound * 2 <= populationSize
          "learningRate": 0,  # co to?
          "discriminativeThreshold": 0.95,
          "weightDecay": 0.1,
          "steps": 15,
          "runs": 1}


class Simulation:

    def __init__(self, parameters=params):
        self.params = parameters
        self.stats = []

    def run(self):

        population = Population(self.params['populationSize'])

        # ds_scores_1 = []
        # ds_scores_2 = []
        # cs_scores_1 = []
        # cs_scores_2 = []

        for step in range(self.params["steps"]):
            print("STEP %d" % step)
            selected_pairs = population.select_pairs_per_round(self.params["gamesPerRound"])
            for speaker, hearer in selected_pairs:
                game = GuessingGame(speaker=speaker, hearer=hearer)
                game.play()

            # ds_score1 = (sum(population.agents[0].ds_scores)/len(population.agents[0].ds_scores)*100)
            # ds_score2 = (sum(population.agents[1].ds_scores) / len(population.agents[1].ds_scores) * 100)
            # cs_score1 = (sum(population.agents[0].cs_scores)/len(population.agents[0].cs_scores)*100)
            # cs_score2 = (sum(population.agents[1].cs_scores) / len(population.agents[1].cs_scores) * 100)
            # print("ds 1: %d%%" % ds_score1)
            # print("ds 2: %d%%" % ds_score2)
            # print("cs 1: %d%%" % cs_score1)
            # print("cs 2: %d%%" % cs_score2)
            # ds_scores_1.append(ds_score1)
            # ds_scores_2.append(ds_score2)
            # cs_scores_1.append(cs_score1)
            # cs_scores_2.append(cs_score2)

        # show
        # x = numpy.arange(1, self.params["steps"]+1)
        # x = range(1, self.params["steps"]+1)
        # plt.ylim(bottom=0)
        # plt.ylim(top=100)
        # plt.xlabel("step")
        # plt.ylabel("success")
        # plt.plot(x, ds_scores_1, '--', x, ds_scores_2, '--', x, cs_scores_1, '-')
        # plt.legend(['line', 'line', 'line', 'line'], loc='best')
        # plt.show()


class RoundStatistics:
    discriminative_success = 0
    guessing_topic_success = 0
    guessing_word_success = 0


sim = Simulation()
sim.run()
