from __future__ import division  # force python 3 division in python 2
import matplotlib.pyplot as plt
from numpy import column_stack
from numpy import linspace
from numpy import arange
from numpy import array
from numpy import zeros
from numpy import amax
from collections import deque
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

line_styles = {0: '-',
               1: ':',
               2: '--',
               3: '-.',
               4: '.',
               5: ',',
               6: 'o'}

class Data:

    def __init__(self, population_size):
        self._population_size_ = population_size
        self._max_weight_ = {a: .0 for a in range(population_size)}
        self.ds_per_agent = [.0 for a in range(population_size)]
        self.cs_scores = deque([0])  # {a: deque([0]) for a in range(population_size)}
        self.matrices = {a: [] for a in range(population_size)}
        self.langs = {a: [] for a in range(population_size)}
        self.cats = {a: [] for a in range(population_size)}
        self.x_left = 0
        self.x_right = 100
        self.x = linspace(self.x_left, self.x_right, 20 * (self.x_right - self.x_left), False)
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

    def store_cats(self, agents):
        for i in range(len(agents)):
            self.cats[i].append([])
            a = agents[i]
            for cat_index in range(len(a.categories)):
                self.cats[i][-1].append([a.categories[cat_index].fun(x_0) for x_0 in self.x])

    def plot_cats(self):
        for i in range(len(self.cats)):
            for step in range(len(self.cats[i])):
                plt.title("categories")
                ax = plt.gca()
                plt.xscale("symlog")
                ax.xaxis.set_major_formatter(ScalarFormatter())
                plt.yscale("symlog")
                ax.yaxis.set_major_formatter(ScalarFormatter())
                colors = sns.color_palette()
                num_of_cats = len(self.cats[i][step])
                for j in range(num_of_cats):
                    plt.plot(self.x, self.cats[i][step][j],
                             color=colors[j % len(colors)], linestyle=line_styles[j // len(colors)],
                             label="%d" % (j + 1))
                plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1, 1))
                plt.tight_layout(pad=0)
                plt.savefig("./simulation_results/cats/categories%d_%d" % (i, step))
                plt.close()

    def store_langs(self, agents):
        for i in range(len(agents)):
            self.langs[i].append([])
            a = agents[i]
            forms_to_categories = {}
            if not a.lxc.size:
                continue
            for f in a.lexicon:
                forms_to_categories[f] = []
            for c in a.categories:
                j = a.categories.index(c)
                m = max(a.lxc[0::, j])
                if m == 0:
                    continue
                else:
                    max_form_indices = [i for i, w in enumerate(a.lxc[0::, j]) if w == m]
                    form = a.lexicon[max_form_indices[0]]
                    forms_to_categories[form].append(j)

            for w in range(len(a.lexicon)):
                f = a.lexicon[w]
                if len(forms_to_categories[f]) == 0:
                    continue
                else:
                    self.langs[i][-1].append([f])
                    for j in forms_to_categories[f]:
                        self.langs[i][-1][-1].append([a.categories[j].fun(x_0) for x_0 in self.x])

    def store_matrices(self, agents):
        for i in range(len(agents)):
            lex = agents[i].lexicon
            lxc = agents[i].lxc
            self.matrices[i].append((list(lex), array(lxc)))
            if lxc.size:
                self._max_weight_[i] = max(self._max_weight_[i], amax(lxc))

    def plot_matrices(self):
        for l in range(len(self.matrices)):
            n_rows = self.matrices[l][-1][1].shape[0]
            n_cols = self.matrices[l][-1][1].shape[1]
            for m in range(len(self.matrices[l])):
                if not self.matrices[l][m][1].size:
                    continue
                n_categories = self.matrices[l][m][1].shape[1]
                n_forms = len(self.matrices[l][m][0])
                #lxc = self.languages[l][m][1].resize((n_rows, n_cols), refcheck=False)
                lxc = zeros((n_rows, n_cols))
                lxc[0:n_forms, 0:n_categories] = self.matrices[l][m][1]
                fig, ax = plt.subplots()
                lxc_ex = column_stack((lxc, linspace(self._max_weight_[l], 0, n_rows)))
                im = ax.imshow(lxc_ex)
                lexicon = self.matrices[l][m][0]
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
                        text = ax.text(j, i, round(lxc_ex[i, j], 3),
                                       ha="center", va="center", color="w")
                for i in range(n_rows):
                    ax.text(n_cols, i, round(lxc_ex[i, n_cols], 2), ha="center", va="center", color="w")

                ax.set_title("Association matrix")
                fig.tight_layout()
                plt.savefig("./simulation_results/matrices/matrix%d_%d" % (l, m))
                plt.close()

    def plot_langs(self):
        for i in range(len(self.langs)):
            # sns.set_palette(colors)
            for step in range(len(self.langs[i])):
                plt.title("language")
                plt.xscale("symlog")
                plt.yscale("symlog")
                ax = plt.gca()
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.yaxis.set_major_formatter(ScalarFormatter())
                colors = sns.color_palette()
                for word_cats_index in range(len(self.langs[i][step])):
                    num_words = len(self.langs[i][step])
                    word_cats = self.langs[i][step][word_cats_index]
                    f = word_cats[0]
                    ls = line_styles[word_cats_index // len(colors)]
                    ci = word_cats_index % len(colors)
                    for y in word_cats[1::]:
                        plt.plot(self.x, y, color=colors[ci], linestyle=ls)
                    plt.plot([], [], color=colors[ci], linestyle=ls, label=f)
                plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1, 1))
                plt.tight_layout(pad=0)
                plt.savefig("./simulation_results/langs/language%d_%d.png" % (i, step))
                plt.close()


class RoundStatistics:
    discriminative_success = 0
    guessing_topic_success = 0
    guessing_word_success = 0
