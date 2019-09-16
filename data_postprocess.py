import argparse
import logging
import pickle
import re
import sys
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from numpy import linspace, zeros, column_stack, arange, log, amax
from pathlib import Path

class PlotCategory:
    def __init__(self):
        x_left = 0
        x_right = 100
        self.plot_space = linspace(x_left, x_right, 20 * (x_right - x_left), False)

    def plot_category(self, agent_index, agent_tuple, step):
        agent = agent_tuple[0]
        plt.title("categories")
        ax = plt.gca()
        plt.xscale("symlog")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        plt.yscale("symlog")
        ax.yaxis.set_major_formatter(ScalarFormatter())

        cats = agent.get_categories()
        linestyles = new_linestyles(cats)

        for cat in cats:
            graph = [cat.fun(x_0) for x_0 in self.plot_space]
            color, linestyle = linestyles[cat]

            plt.plot(self.plot_space, graph,
                     color=color,
                     linestyle=linestyle,
                     label="%d" % (cat.id + 1))

        plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1, 1))
        plt.tight_layout(pad=0)

        plt.savefig("./simulation_results/cats/categories%d_%d" % (agent_index, step))
        plt.close()


def new_linestyles(seq):
    linestyles = [(color, style) for style in ['solid', 'dotted', 'dashed', 'dashdot'] for color in sns.color_palette()]
    return dict(zip(seq, linestyles))


class PlotLanguage:
    def __init__(self):
        x_left = 0
        x_right = 100
        self.plot_space = linspace(x_left, x_right, 20 * (x_right - x_left), False)

    def plot_language(self, agent_index, agent_tuple, step):
        agent = agent_tuple[0]
        categories = agent.get_categories()
        forms_to_categories = dict()

        for (category_index, _) in enumerate(categories):
            word_sorted_by_val = agent.language.get_words_sorted_by_val(category_index, threshold=0)
            if len(word_sorted_by_val) > 0:
                word = word_sorted_by_val[0]

                if word not in forms_to_categories:
                    forms_to_categories[word] = []

                category_connected = categories[category_index]
                forms_to_categories[word].append(category_connected)

        plt.title("language in step {} of agent {}".format(step, agent_index))
        plt.xscale("symlog")
        plt.yscale("symlog")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())

        word2linestyles = new_linestyles(agent.get_lexicon())
        for form, categories in forms_to_categories.items():
            color, line = word2linestyles[form]
            for category in categories:
                plt.plot(self.plot_space, map(category.fun, self.plot_space), color=color, linestyle=line)
            plt.plot([], [], color=color, linestyle=line, label=form)

        plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1, 1))
        plt.tight_layout(pad=0)
        plt.savefig("./simulation_results/langs/language%d_%d.png" % (agent_index, step))
        plt.close()


class PlotLanguage2:
    def __init__(self):
        x_left = 0
        x_right = 100
        self.plot_space = linspace(x_left, x_right, 20 * (x_right - x_left), False)

    def plot_language(self, agent_index, agent_tuple, step):
        agent = agent_tuple[0]
        lexicon = agent.get_lexicon()

        lang = []

        for word in lexicon:
            category2sth = zip(agent.get_categories(), agent.get_categories_by_word(word))
            fy = [sum([cat.fun(x) * wei for cat, wei in category2sth]) for x in self.plot_space]
            lang.append([word, fy])

        plt.title("language2 in step {} of agent {}".format(step, agent_index))
        plt.xscale("symlog")
        plt.yscale("symlog")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        word2linestyles = new_linestyles(lexicon)
        for form, y in lang:
            color, linestyle = word2linestyles[form]
            plt.plot(self.plot_space, y, color=color, linestyle=linestyle)
            plt.plot([], [], color=color, linestyle=linestyle, label=form)
        plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1, 1))
        plt.tight_layout(pad=0)
        plt.savefig("./simulation_results/langs2/language%d_%d.png" % (agent_index, step))
        # if not self.pickle_mode else self.step - self.pickle_step + step + 1))
        plt.close()


class PlotMatrix:
    def plot_matrix(self, agent_index, agent_tuple, step):
        agent = agent_tuple[0]
        agent_last = agent_tuple[1]
        matrix = agent.language.lxc.to_array()
        max_shape = agent_last.language.lxc.max_shape()
        lang = list(agent.get_lexicon())
        cats = [c.id for c in agent.language.categories]

        # self._shape_[i] = (max(self._shape_[i][0], lxc.shape[0]), max(self._shape_[i][1], lxc.shape[1]))
        # self.matrices[i].append((list(lex), array(lxc), cats))
        # matrix = (list(lex), array(lxc), cats)
        # if not matrix.size:
        #     return

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
        output_name = "./simulation_results/matrices/matrix%d_%d" % (agent_index, step)
        plt.savefig(output_name)
        plt.close()


class PlotSuccess:
    def plot_success(self, population, step, dt):

        # _ds = population.ds[0:step + 1]
        # _cs = population.cs[0:step + 1]

        x = range(1, step + 1)
        plt.ylim(bottom=0)
        plt.ylim(top=100)
        plt.xlabel("step")
        plt.ylabel("success")
        x_ex = range(0, step + 3)
        th = [dt * 100 for i in x_ex]
        plt.plot(x_ex, th, ':', linewidth=0.2)
        plt.plot(x, population.ds, '--')
        plt.plot(x, population.cs, '-')
        plt.legend(['dt', 'ds', 'gg1s'], loc='best')
        plt.savefig("./simulation_results/success.pdf")
        plt.close()


class CommandExecutor:
    def __init__(self):
        self.agent_commands = []
        self.population_commands = []

    def add_agent_command(self, command):
        self.agent_commands.append(command)

    def add_population_command(self, command):
        self.population_commands.append(command)

    def execute_commands(self, params, population, step):
        for command_exec in self.population_commands:
            command_exec(params, population, step)

        for agent_index, agent_tuple in enumerate(zip(population, last_population)):
            # assert that zip between agent at current and last step is valid
            assert agent_tuple[0].id == agent_tuple[1].id
            for command_exec in self.agent_commands:
                if agent_tuple[0].language.lxc.size():
                    command_exec(agent_index, agent_tuple, step)


class PathProvider:
    def __init__(self, root):
        self.root = Path(root)

    def get_sorted_paths(self):
        data_paths = list(self.root.glob('step[0-9]*.p'))
        return sorted(data_paths, cmp=self.cmp)

    # compare files stepA.p, stepB.p by its numerals A, B
    def cmp(self, path1, path2):
        path1no = int(re.search('step(\d+)\.p', path1.name).group(1))
        path2no = int(re.search('step(\d+)\.p', path2.name).group(1))
        return path1no - path2no


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    parser = argparse.ArgumentParser(prog='plotting data')

    parser.add_argument('--data_path', '-d', help='pickeled input data path', type=str,
                        default=".\simulation_results\data")
    parser.add_argument('--plot_cats', '-c', help='plot categories', type=bool, default=False)
    parser.add_argument('--plot_langs', '-l', help='plot languages', type=bool, default=False)
    parser.add_argument('--plot_langs2', '-l2', help='plot languages 2', type=bool, default=False)
    parser.add_argument('--plot_matrices', '-m', help='plot matrices', type=bool, default=False)
    parser.add_argument('--plot_success', '-s', help='plot success', type=bool, default=True)

    parsed_params = vars(parser.parse_args())

    logging.debug("loading pickled simulation from %s file", parsed_params['data_path'])

    # set commands to be executed
    command_executor = CommandExecutor()

    if parsed_params['plot_cats']:
        command_executor.add_agent_command(PlotCategory().plot_category)
    if parsed_params['plot_langs']:
        command_executor.add_agent_command(PlotLanguage().plot_language)
    if parsed_params['plot_langs2']:
        command_executor.add_agent_command(PlotLanguage2().plot_language)
    if parsed_params['plot_matrices']:
        command_executor.add_agent_command(PlotMatrix().plot_matrix)


    data_paths = PathProvider(parsed_params['data_path']).get_sorted_paths()
    data_last_step_path = data_paths[-1]
    params, last_step, last_population = pickle.load(data_last_step_path.open('rb'))

    if parsed_params['plot_success']:
        PlotSuccess().plot_success(last_population, last_step, params['discriminative_threshold'])

    start_time = time.time()

    logging.debug('execution time %dsec, with params %s', time.time() - start_time, parsed_params)

    for path in data_paths:

        params, step, population = pickle.load(path.open('rb'))

        for agent_index, agent_tuple in enumerate(zip(population, last_population)):
            # assert that zip between agent at current and last step is valid
            assert agent_tuple[0].id == agent_tuple[1].id
            command_executor.execute_commands(agent_index, agent_tuple, step)


    logging.debug('execution time %dsec, with params %s', time.time() - start_time, parsed_params)
