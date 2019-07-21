from __future__ import division  # force python 3 division in python 2
import logging,sys
import matplotlib.pyplot as plt
from agent import Population
from guessing_game import GuessingGame
from language import Language
from data import Data, ScoreCalculator
# import cProfile

params = {"population_size": 2,
          "games_per_round": 1,  # gamesPerRound * 2 <= populationSize
          "learning_rate": 0,  # co to?
          "discriminative_threshold": 0.95,
          "weight_decay": 0.1,
          "steps": 30,
          "runs": 1}


class Simulation:

    def __init__(self, parameters=params):
        self.params = parameters
        self.data = Data(self.params['population_size'])

    def run(self):

        population = Population(self.params['population_size'])
        scores_calculator_0 = ScoreCalculator()
        scores_calculator_1 = ScoreCalculator()

        for step in range(self.params["steps"]):
            logging.debug("--\nSTEP %d" % step)
            selected_pairs = population.select_pairs_per_round(self.params["games_per_round"])
            for speaker, hearer in selected_pairs:
                game = GuessingGame(speaker=speaker, hearer=hearer)
                game.play()

                logging.debug("Number of categories of Agent(%d): %d" % (population.agents[0].id,
                                                                         len(population.agents[0].categories)))
                super(Language, population.agents[0]).plot("./simulation_results/cats/categories%d_%d" % (0, step))
                logging.debug("Number of categories of Agent(%d): %d" % (population.agents[1].id,
                                                                         len(population.agents[1].categories)))
                super(Language, population.agents[1]).plot("./simulation_results/cats/categories%d_%d" % (1, step))
            scores_calculator_0.update_result(population.agents[0])
            scores_calculator_1.update_result(population.agents[1])

            #store languages
            self.data.store_languages(population.agents)

        self.data.plot_languages()
        x = range(1, self.params["steps"] + 1)
        plt.ylim(bottom=0)
        plt.ylim(top=100)
        plt.xlabel("step")
        plt.ylabel("success")
        plt.plot(x, scores_calculator_0.ds_scores, '--', x, scores_calculator_1.ds_scores, '--', x,
                 scores_calculator_0.cs_scores, '-')
        plt.legend(['line', 'line', 'line', 'line'], loc='best')
        # plt.show()
        plt.savefig("./simulation_results/success.pdf")
        plt.close()
        # plot language of the first agent
        print("plotting languages")
        population.agents[0].plot(filename="./simulation_results/language0.pdf")
        # population.agents[0].plot_categories()
        population.agents[1].plot(filename="./simulation_results/language1.pdf")


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
sim = Simulation()
sim.run()
