from __future__ import division  # force python 3 division in python 2
import logging, sys
import matplotlib.pyplot as plt
from agent import Population
from guessing_game import GuessingGame
from language import Language
from data import Data
# import cProfile

params = {"population_size": 10,
          "learning_rate": 0,  # co to?
          "discriminative_threshold": 0.95,
          "weight_decay": 0.1,
          "steps": 15,
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
                result = game.play()

                logging.debug("Number of categories of Agent(%d): %d" % (speaker.id, len(speaker.categories)))
                super(Language, speaker).plot("./simulation_results/cats/categories%d_%d" % (speaker.id, step))
                logging.debug("Number of categories of Agent(%d): %d" % (hearer.id, len(hearer.categories)))
                super(Language, hearer).plot("./simulation_results/cats/categories%d_%d" % (hearer.id, step))
                speaker.plot(filename="./simulation_results/langs/language%d_%d.png" % (speaker.id, step))
                hearer.plot(filename="./simulation_results/langs/language%d_%d.png" % (hearer.id, step))
                self.data.store_ds_result(speaker.id, speaker.discriminative_success)
                self.data.store_ds_result(hearer.id, hearer.discriminative_success)
                self.data.store_cs_result(result)

            self.data.store_languages(population.agents)
            ds.append(self.data.get_ds())
            cs.append(self.data.get_cs())

        self.data.plot_languages()
        x = range(1, self.params["steps"] + 1)
        plt.ylim(bottom=0)
        plt.ylim(top=100)
        plt.xlabel("step")
        plt.ylabel("success")
        plt.plot(x, ds, '--', label="discriminative success")
        plt.plot(x, cs, '-', label="communicative success")
        plt.legend(['line', 'line'], loc='best')
        # # plt.show()
        plt.savefig("./simulation_results/success.pdf")
        plt.close()


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

sim = Simulation()
sim.run()
