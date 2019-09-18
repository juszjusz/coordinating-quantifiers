from __future__ import division  # force python 3 division in python 2
import matplotlib
matplotlib.use('Agg')

import argparse
import logging
import sys
import time
from numpy import column_stack
from numpy import linspace
from numpy import arange
from numpy import array
from numpy import zeros
from numpy import amax
from numpy import log
from pathlib import Path
import seaborn as sns
import pickle
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


class Data:

    def __init__(self, population_size, pickle_mode=True):
        self.pickle_mode = pickle_mode
        self._population_size_ = population_size
        # self._max_weight_ = {a: .0 for a in range(population_size)}
        self.ds_per_agent = [.0 for a in range(population_size)]
        self.cs_per_agent = [.0 for a in range(population_size)]
        self.matrices = {a: [] for a in range(population_size)}
        self.langs = {a: [] for a in range(population_size)}
        self.langs2 = {a: [] for a in range(population_size)}
        self.cats = {a: [] for a in range(population_size)}
        self.x_left = 0
        self.x_right = 100
        self.x = linspace(self.x_left, self.x_right, 20 * (self.x_right - self.x_left), False)
        self.step = 0
        self.pickle_step = 10
        self.pickle_count = 0
        self._shape_ = {a: (0, 0) for a in range(population_size)}
        self.ds = []
        self.cs = []
        self.cat_linestyles = {a: deque([(c, s) for s in ['solid', 'dotted', 'dashed', 'dashdot'] for c in sns.color_palette()]) for a in range(population_size)}
        self.cats_to_linestyles = {a: {} for a in range(population_size)}
        self.word_linestyles = {a: deque([(c, s) for s in ['solid', 'dotted', 'dashed', 'dashdot'] for c in sns.color_palette()]) for a in range(population_size)}
        self.words_to_linestyles = {a: {} for a in range(population_size)}

    def pickle(self, step, agents):
        self.step = step
        if self.pickle_mode and (step + 1) % self.pickle_step == 0:
            pickle.dump(self._shape_, open("./simulation_results/data/info.p", "wb"), protocol=2)
            pickle.dump(self, open("./simulation_results/data/%d.p" % self.pickle_count, "wb"), protocol=2)
            self.pickle_count = self.pickle_count + 1
            self.matrices = {a: [] for a in range(self._population_size_)}
            self.langs = {a: [] for a in range(self._population_size_)}
            self.langs2 = {a: [] for a in range(self._population_size_)}
            self.cats = {a: [] for a in range(self._population_size_)}

    def store_ds(self, agents):
        for i in range(self._population_size_):
            self.ds_per_agent[i] = agents[i].get_discriminative_success()*100
        self.ds.append(self.get_ds())

    def store_cs(self, agents):
        for i in range(self._population_size_):
            self.cs_per_agent[i] = agents[i].get_communicative_success()
        self.cs.append(self.get_cs())

    def get_ds(self):
        return sum(self.ds_per_agent) / self._population_size_

    def get_cs(self):
        return sum(self.cs_per_agent) / self._population_size_

    def store_cats(self, agents):
        for i in range(len(agents)):
            self.cats[i].append([])
            a = agents[i]
            current_cats = set([c.id for c in a.get_categories()])
            previous_cats = set(self.cats_to_linestyles[i].keys())
            forgotten_cats = previous_cats - current_cats
            new_cats = current_cats - previous_cats

            for fc in forgotten_cats:
                self.cat_linestyles[i].append(self.cats_to_linestyles[i][fc])
                self.cats_to_linestyles[i].pop(fc)

            for nc in new_cats:
                self.cats_to_linestyles[i][nc] = self.cat_linestyles[i].popleft()

            for cat_index in range(len(a.get_categories())):
                self.cats[i][-1].append(
                    (a.get_categories()[cat_index].id,
                     [a.get_categories()[cat_index].fun(x_0) for x_0 in self.x],
                     self.cats_to_linestyles[i][a.get_categories()[cat_index].id]))

    def plot_cats(self):
        # multiprocessing.Pool - python3
        # with multiprocessing.Pool() as executor:
        #     executor.map(self.plot_cat, [category_index for category_index in range(len(self.cats))])
        for category_index in range(len(self.cats)):
            self.plot_cat(category_index)

    def plot_cat(self, category_index):
        cat = self.cats[category_index]
        for step in range(len(cat)):
            plt.title("categories")
            ax = plt.gca()
            plt.xscale("symlog")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            plt.yscale("symlog")
            ax.yaxis.set_major_formatter(ScalarFormatter())
            num_of_cats = len(cat[step])
            for j in range(num_of_cats):
                cat_id = cat[step][j][0]
                cat_y = cat[step][j][1]
                plt.plot(self.x, cat_y,
                         color=cat[step][j][2][0],
                         linestyle=cat[step][j][2][1],
                         label="%d" % (cat_id + 1))
            plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1, 1))
            plt.tight_layout(pad=0)

            plt.savefig("./simulation_results/cats/categories%d_%d" % (
                category_index, step if not self.pickle_mode else self.step - self.pickle_step + step + 1))
            plt.close()

    def store_langs(self, agents):
        for i in range(len(agents)):
            self.langs[i].append([])
            self.langs2[i].append([])
            a = agents[i]

            current_words = set(a.get_lexicon())
            previous_words = set(self.words_to_linestyles[i].keys())
            forgotten_words = previous_words - current_words
            new_words = current_words - previous_words

            for fw in forgotten_words:
                self.word_linestyles[i].append(self.words_to_linestyles[i][fw])
                self.words_to_linestyles[i].pop(fw)

            for nw in new_words:
                self.words_to_linestyles[i][nw] = self.word_linestyles[i].popleft()

            #  lang
            forms_to_categories = {}

            if not a.language.lxc.size():
                continue
            for f in a.get_lexicon():
                forms_to_categories[f] = []
            for c in a.get_categories():
                j = a.get_categories().index(c)
                m = max(a.language.get_words_by_category(j))
                if m == 0:
                    continue
                else:
                    max_form_indices = [ind for ind, w in enumerate(a.language.get_words_by_category(j)) if w == m]
                    form = a.get_lexicon()[max_form_indices[0]]
                    forms_to_categories[form].append(j)
            for w in range(len(a.get_lexicon())):
                f = a.get_lexicon()[w]
                if len(forms_to_categories[f]) == 0:
                    continue
                else:
                    self.langs[i][-1].append([f])
                    self.langs[i][-1][-1].append(self.words_to_linestyles[i][f])
                    for j in forms_to_categories[f]:
                        self.langs[i][-1][-1].append([a.get_categories()[j].fun(x_0) for x_0 in self.x])

            #  lang2

            if not a.language.lxc.size():
                continue
            for w in range(len(a.get_lexicon())):
                f = a.get_lexicon()[w]
                self.langs2[i][-1].append([f])
                fy = [sum([cat.fun(x)*wei for cat, wei in zip(a.get_categories(), a.language.lxc.to_matrix()[w])]) for x in self.x]
                self.langs2[i][-1][-1].append(fy)
                self.langs2[i][-1][-1].append(self.words_to_linestyles[i][f])

    def store_matrices(self, agents):
        for i in range(len(agents)):
            lex = agents[i].get_lexicon()
            lxc = agents[i].language.lxc.to_matrix()
            ids = [c.id for c in agents[i].language.categories]
            self._shape_[i] = (max(self._shape_[i][0], lxc.shape[0]), max(self._shape_[i][1], lxc.shape[1]))
            self.matrices[i].append((list(lex), array(lxc), ids))
            # if lxc.size:
            #     self._max_weight_[i] = max(self._max_weight_[i], amax(lxc))

    def plot_matrices(self, max_shapes):
        for agent in range(self._population_size_):
            max_shape = max_shapes[agent]
            agent_matrices = self.matrices[agent]
            for simulation_step in range(len(agent_matrices)):
                output_name = "./simulation_results/matrices/matrix%d_%d" % (
                    agent,
                    simulation_step if not self.pickle_mode else self.step - self.pickle_step + simulation_step + 1
                )
                lang, matrix, cats = agent_matrices[simulation_step]
                self.plot_matrix(lang, matrix, cats, max_shape, output_name)

    @staticmethod
    def plot_matrix(lang, matrix, cats, max_shape, output_name):
        if not matrix.size:
            return
        n_rows = max_shape[0]
        n_cols = max_shape[1]
        n_categories = matrix.shape[1]
        n_forms = len(lang)
        # lxc = self.languages[l][m][1].resize((n_rows, n_cols), refcheck=False)
        lxc = zeros(max_shape)
        lxc[0:n_forms, 0:n_categories] = matrix
        fig, ax = plt.subplots()
        lxc_ex = column_stack((lxc, linspace(amax(lxc), 0, n_rows)))
        lxc_ex_log = log(lxc_ex + 1.)
        im = ax.imshow(lxc_ex_log, aspect='auto')
        lexicon = lang
        # We want to show all ticks...
        ax.set_xticks(arange(n_cols + 1))
        ax.set_yticks(arange(n_rows))
        # ... and label them with the respective list entries
        x_tick_labels = [str(cid + 1) for cid in cats]
        for _ in range(n_categories, n_cols):
            x_tick_labels.append('-')
        x_tick_labels.append("s")
        ax.set_xticklabels(x_tick_labels, fontdict={'fontsize': 8})
        for _ in range(len(lexicon), n_rows):
            x_tick_labels.append('-')
        ax.set_yticklabels(lexicon, fontdict={'fontsize': 8})
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        # for i in range(len(lexicon)):
        #     for j in range(n_categories):
        #         text = ax.text(j, i, round(lxc_ex[i, j], 3),
        #                        ha="center", va="center", color="w")
        # for i in range(n_rows):
        #     ax.text(n_cols, i, round(lxc_ex[i, n_cols], 2), ha="center", va="center", color="w")

        ax.set_title("Association matrix")
        fig.tight_layout()
        plt.savefig(output_name)
        plt.close()

    def plot_langs(self):
        # multiprocessing.Pool - python3
        # with multiprocessing.Pool() as executor:
        #     executor.map(self.plot_lang, [lang_index for lang_index in range(len(self.langs))])
        for lang_index in range(len(self.langs)):
            self.plot_lang(lang_index)

    def plot_langs2(self):
        # print("plot langs 2")
        # multiprocessing.Pool - python3
        # with multiprocessing.Pool() as executor:
        #     executor.map(self.plot_lang, [lang_index for lang_index in range(len(self.langs))])
        for lang_index in range(len(self.langs2)):
            self.plot_lang2(lang_index)

    def plot_lang(self, lang_index):
        lang = self.langs[lang_index]
        for step in range(len(lang)):
            plt.title("language")
            plt.xscale("symlog")
            plt.yscale("symlog")
            ax = plt.gca()
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter())
            colors = sns.color_palette()
            for word_cats_index in range(len(self.langs[lang_index][step])):
                #num_words = len(self.langs[lang_index][step])
                word_cats = self.langs[lang_index][step][word_cats_index]
                f = word_cats[0]
                ls = word_cats[1]
                #ls = self.words_to_linestyles[lang_index][f]
                #ci = word_cats_index % len(colors)
                for y in word_cats[2::]:
                    plt.plot(self.x, y, color=ls[0], linestyle=ls[1])
                plt.plot([], [], color=ls[0], linestyle=ls[1], label=f)
            plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1, 1))
            plt.tight_layout(pad=0)
            plt.savefig("./simulation_results/langs/language%d_%d.png" % (
                lang_index, step if not self.pickle_mode else self.step - self.pickle_step + step + 1))
            plt.close()

    def plot_lang2(self, lang_index):
        lang = self.langs2[lang_index]
        for step in range(len(lang)):
            plt.title("language2")
            plt.xscale("symlog")
            plt.yscale("symlog")
            ax = plt.gca()
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter())
            #colors = sns.color_palette()
            for fys_ind in range(len(self.langs2[lang_index][step])):
                #num_words = len(self.langs2[lang_index][step])
                fys = self.langs2[lang_index][step][fys_ind]
                f = fys[0]
                y = fys[1]
                s = fys[2]
                # print("Agent %d, word %s" % (lang_index, f))
                #ls = self.words_to_linestyles[lang_index][f]
                plt.plot(self.x, y, color=s[0], linestyle=s[1])
                plt.plot([], [], color=s[0], linestyle=s[1], label=f)
            plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1, 1))
            plt.tight_layout(pad=0)
            plt.savefig("./simulation_results/langs2/language%d_%d.png" % (
                lang_index, step if not self.pickle_mode else self.step - self.pickle_step + step + 1))
            plt.close()

    def plot_success(self, dt, step):
        x = range(1, step + 2)
        plt.ylim(bottom=0)
        plt.ylim(top=100)
        plt.xlabel("step")
        plt.ylabel("success")
        x_ex = range(0, step + 3)
        th = [dt*100 for i in x_ex]
        plt.plot(x_ex, th, ':', linewidth=0.2)
        plt.plot(x, self.ds, '--')
        plt.plot(x, self.cs, '-')
        plt.legend(['dt', 'ds', 'gg1s'], loc='best')
        plt.savefig("./simulation_results/success.pdf")
        plt.close()


class RoundStatistics:
    discriminative_success = 0
    guessing_topic_success = 0
    guessing_word_success = 0


class DataPostprocessor:
    def __init__(self, root="./simulation_results/data"):
        self.root = Path(root)
        self.commands = []

    def add_command(self, command):
        self.commands.append(command)

    def execute_commands(self):
        max_shape = pickle.load(self.root.joinpath("info.p").open("rb"))
        data_paths = self.root.glob("[0-9]*.p")
        data_unpickled = (pickle.load(data_path.open('rb')) for data_path in data_paths)
        for data in data_unpickled:
            for command_exec in self.commands:
                command_exec({'data': data, 'max_shape': max_shape})

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    parser = argparse.ArgumentParser(prog='plotting data')

    parser.add_argument('--data_path', '-d', help='pickeled input data path', type=str,
                        default="./simulation_results/data/%d.p")
    parser.add_argument('--plot_cats', '-c', help='plot categories', type=bool, default=True)
    parser.add_argument('--plot_langs', '-l', help='plot languages', type=bool, default=True)
    parser.add_argument('--plot_langs2', '-l2', help='plot languages 2', type=bool, default=True)
    parser.add_argument('--plot_matrices', '-m', help='plot matrices', type=bool, default=True)

    parsed_params = vars(parser.parse_args())

    start_time = time.time()
    data_provider = DataPostprocessor()
    if parsed_params['plot_cats']:
        data_provider.add_command(lambda x: x['data'].plot_cats())
    if parsed_params['plot_langs']:
        data_provider.add_command(lambda x: x['data'].plot_langs())
    if parsed_params['plot_langs2']:
        data_provider.add_command(lambda x: x['data'].plot_langs2())
    if parsed_params['plot_matrices']:
        data_provider.add_command(lambda x: x['data'].plot_matrices(x['max_shape']))
    data_provider.execute_commands()
    logging.debug('execution time %dsec, with params %s', time.time() - start_time, parsed_params)
