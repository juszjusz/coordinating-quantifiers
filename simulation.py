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
        ds = []
        cs = []

        for step in range(self.params["steps"]):
            logging.debug("\n------------\nSTEP %d" % step)
            selected_pairs = population.select_pairs_per_round(self.params['population_size']//2)
            for speaker, hearer in selected_pairs:
                game = GuessingGame(speaker=speaker, hearer=hearer)
                logging.debug("\nGAME(%d, %d)" % (speaker.id, hearer.id))
                result = game.play(speaker, hearer)

                logging.debug("Number of categories of Agent(%d): %d" % (speaker.id, len(speaker.categories)))
                logging.debug("Number of categories of Agent(%d): %d" % (hearer.id, len(hearer.categories)))
                self.data.store_ds_result(speaker.id, speaker.discriminative_success)
                self.data.store_ds_result(hearer.id, hearer.discriminative_success)
                self.data.store_cs_result(result)

            self.data.store_matrices(population.agents)
            self.data.store_langs(population.agents)
            self.data.store_cats(population.agents)
            ds.append(self.data.get_ds())
            cs.append(self.data.get_cs())

            x = range(1, step + 2)
            plt.ylim(bottom=0)
            plt.ylim(top=100)
            plt.xlabel("step")
            plt.ylabel("success")
            x_ex = range(0, step + 3)
            th = [95 for i in x_ex]
            plt.plot(x_ex, th, ':', linewidth=0.2)
            plt.plot(x, ds, '--', label="discriminative success")
            plt.plot(x, cs, '-', label="communicative success")
            plt.legend(['line', 'line'], loc='best')
            # # plt.show()
            plt.savefig("./simulation_results/success.pdf")
            plt.close()

        self.data.plot_matrices()  # saves language matrices to ./simulation_results/matrices
        self.data.plot_langs()  # saves languages to ./simulation_results/langs
        self.data.plot_cats()  # saves categories to ./simulation_results/cats


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

sim = Simulation()
sim.run()
