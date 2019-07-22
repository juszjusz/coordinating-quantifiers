from __future__ import division  # force python 3 division in python 2
import matplotlib.pyplot as plt
from numpy import column_stack
from numpy import linspace
from numpy import arange
from numpy import array
from numpy import zeros
from numpy import amax
from collections import deque


class Data:

    def __init__(self, population_size):
        self._population_size_ = population_size
        self._max_weight_ = {a: .0 for a in range(population_size)}
        self.ds_per_agent = [.0 for a in range(population_size)]
        self.cs_scores = deque([0])  # {a: deque([0]) for a in range(population_size)}
        self.languages = {a: [] for a in range(population_size)}
        # self.ds = []
        # self.cs = []

    def store_ds_result(self, agent_index, ds):
        self.ds_per_agent[agent_index] = ds

    def store_cs_result(self, cs_result):
        if len(self.cs_scores) == 50:
            self.cs_scores.rotate(-1)
            self.cs_scores[-1] = int(cs_result)
        else:
            self.cs_scores.append(int(cs_result))

    def get_ds(self):
        return sum(self.ds_per_agent)/self._population_size_

    def get_cs(self):
        return sum(self.cs_scores) / len(self.cs_scores) * 100

    def store_languages(self, agents):
        for i in range(len(agents)):
            lex = agents[i].lexicon
            lxc = agents[i].lxc
            self.languages[i].append((list(lex), array(lxc)))
            if lxc.size:
                self._max_weight_[i] = max(self._max_weight_[i], amax(lxc))

    def plot_languages(self):
        for l in range(len(self.languages)):
            n_rows = self.languages[l][-1][1].shape[0]
            n_cols = self.languages[l][-1][1].shape[1]
            for m in range(len(self.languages[l])):
                if not self.languages[l][m][1].size:
                    continue
                n_categories = self.languages[l][m][1].shape[1]
                n_forms = len(self.languages[l][m][0])
                #lxc = self.languages[l][m][1].resize((n_rows, n_cols), refcheck=False)
                lxc = zeros((n_rows, n_cols))
                lxc[0:n_forms, 0:n_categories] = self.languages[l][m][1]
                fig, ax = plt.subplots()
                lxc_ex = column_stack((lxc, linspace(self._max_weight_[l], 0, n_rows)))
                im = ax.imshow(lxc_ex)
                lexicon = self.languages[l][m][0]
                # We want to show all ticks...
                ax.set_xticks(arange(n_cols + 1))
                ax.set_yticks(arange(n_rows))
                # ... and label them with the respective list entries
                x_tick_labels = [str(j + 1) for j in arange(n_categories)]
                for t in range(n_categories,n_cols):
                    x_tick_labels.append('-')
                x_tick_labels.append("scale")
                ax.set_xticklabels(x_tick_labels)
                for t in range(len(lexicon), n_rows):
                    x_tick_labels.append('-')
                ax.set_yticklabels(lexicon)
                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
                         rotation_mode="anchor")
                for i in range(len(lexicon)):
                    for j in range(n_categories):
                        text = ax.text(j, i, round(lxc_ex[i, j], 2),
                                       ha="center", va="center", color="w")
                for i in range(n_rows):
                    ax.text(n_cols, i, round(lxc_ex[i, n_cols], 2), ha="center", va="center", color="w")

                ax.set_title("Association matrix")
                fig.tight_layout()
                plt.savefig("./simulation_results/lxc/lxc%d_%d" % (l,m))
                plt.close()


class RoundStatistics:
    discriminative_success = 0
    guessing_topic_success = 0
    guessing_word_success = 0
