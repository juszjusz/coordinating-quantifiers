from __future__ import division  # force python 3 division in python 2
import logging, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from agent import Population
from guessing_game import GuessingGame
from language import Language
from data import Data
# import cProfile

params = {"population_size": int(sys.argv[1]),
          "learning_rate": 0,  # co to?
          "discriminative_threshold": 0.95,
          "weight_decay": 0.1,
          "steps": int(sys.argv[2]),
          "runs": 1}


class Simulation:

    def __init__(self, parameters=params):
        self.params = parameters
        self.data = Data(self.params['population_size'])

    def run(self):

        population = Population(self.params['population_size'])

        for step in range(self.params["steps"]):
            logging.debug("\n------------\nSTEP %d" % step)
            selected_pairs = population.select_pairs_per_round(self.params['population_size']//2)

            for speaker, hearer in selected_pairs:
                game = GuessingGame()
                logging.debug("\nGAME(%d, %d)" % (speaker.id, hearer.id))
                result = game.play(speaker=speaker, hearer=hearer)

                logging.debug("Number of categories of Agent(%d): %d" % (speaker.id, len(speaker.get_categories())))
                logging.debug("Number of categories of Agent(%d): %d" % (hearer.id, len(hearer.get_categories())))
                self.data.store_cs(result)

            self.data.store_ds(population.agents)
            self.data.store_matrices(population.agents)
            self.data.store_langs(population.agents)
            self.data.store_cats(population.agents)
            self.data.pickle(step, population.agents)
            self.data.plot_success(step)


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

sim = Simulation()
sim.run()
