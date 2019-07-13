from __future__ import division # force python 3 division in python 2
import logging, sys
# import matplotlib as plt
# matplotlib.use("Agg")
from agent import Population
from guessing_game import GuessingGame
# import cProfile


params = {"populationSize": 2,
          "gamesPerRound": 1,  # gamesPerRound * 2 <= populationSize
          "learningRate": 0,  # co to?
          "discriminativeThreshold": 0.95,
          "weightDecay": 0.1,
          "steps": 60,
          "runs": 1}


class Simulation:

    def __init__(self, parameters=params):
        self.params = parameters
        self.stats = []

    def run(self):

        population = Population(self.params['populationSize'])

        ds_scores_1 = []
        ds_scores_2 = []
        cs_scores_1 = []

        for step in range(self.params["steps"]):
            logging.debug("--\nSTEP %d" % step)
            selected_pairs = population.select_pairs_per_round(self.params["gamesPerRound"])
            for speaker, hearer in selected_pairs:
                game = GuessingGame(speaker=speaker, hearer=hearer)
                game.play()
                population.agents[0].plot_categories("./cats/categories%d_%d" % (0, step))
                population.agents[1].plot_categories("./cats/categories%d_%d" % (1, step))

            # ds_score1 = (sum(population.agents[0].ds_scores)/len(population.agents[0].ds_scores)*100)
            # ds_score2 = (sum(population.agents[1].ds_scores) / len(population.agents[1].ds_scores) * 100)
            # cs_score1 = (sum(population.agents[0].cs_scores)/len(population.agents[0].cs_scores)*100)
            # ds_scores_1.append(ds_score1)
            # ds_scores_2.append(ds_score2)
            # cs_scores_1.append(cs_score1)

        # x = range(1, self.params["steps"]+1)
        # plt.ylim(bottom=0)
        # plt.ylim(top=100)
        # plt.xlabel("step")
        # plt.ylabel("success")
        # plt.plot(x, ds_scores_1, '--', x, ds_scores_2, '--', x, cs_scores_1, '-')
        # plt.legend(['line', 'line', 'line', 'line'], loc='best')
        # plt.show()
        # plt.savefig("success.pdf")
        # plot language of the first agent
        # population.agents[0].plot_bottom_up()
        # population.agents[0].plot_categories()
        # population.agents[1].plot_categories()


class RoundStatistics:
    discriminative_success = 0
    guessing_topic_success = 0
    guessing_word_success = 0


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
sim = Simulation()
sim.run()
