import os

import matplotlib
from pathlib import Path

from path_provider import PathProvider
from stats import confidence_intervals, means

matplotlib.use('Agg')

import argparse
import logging
import pickle
import sys
import time

from multiprocessing import Process
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from numpy import linspace, zeros, column_stack, arange, log, amax, zeros, mean, std
from stimulus import StimulusFactory
from scipy import stats
from math import sqrt


class PlotCategoryCommand:
    def __init__(self, categories_path, params):
        x_left = 0
        x_right = 1.1 if params['stimulus'] == 'quotient' else params['max_num'] + 1
        self.plot_space = linspace(x_left, x_right, 20 * (x_right - x_left), False)
        self.categories_path = categories_path

    def __call__(self, agent_index, agent_tuple, step):
        agent = agent_tuple[0]
        plt.title("categories")
        ax = plt.gca()
        #plt.xscale("symlog")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        #plt.yscale("symlog")
        ax.yaxis.set_major_formatter(ScalarFormatter())

        cats = agent.get_categories()
        linestyles = new_linestyles(cats)

        for cat in cats:
            graph = [cat.fun(x_0) for x_0 in self.plot_space]
            color, linestyle = linestyles[cat]

            plt.plot(self.plot_space, graph,
                     color=color,
                     linestyle=linestyle,
                     label="%d" % (cat.id))

        plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1, 1))
        plt.tight_layout(pad=0)
        new_path = self.categories_path.joinpath('categories{}_{}.png'.format(agent_index, step))
        plt.savefig(str(new_path))
        plt.close()


def new_linestyles(seq):
    linestyles = [(color, style) for style in ['solid', 'dotted', 'dashed', 'dashdot'] for color in sns.color_palette()]
    return dict(zip(seq, linestyles))


class PlotLanguageCommand:
    def __init__(self, lang_path, params):
        x_left = 0
        x_right = 1.1 if params['stimulus'] == 'quotient' else params['max_num'] + 1
        self.plot_space = linspace(x_left, x_right, 20 * (x_right - x_left), False)
        self.lang_path = lang_path

    def __call__(self, agent_index, agent_tuple, step):
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
        # plt.xscale("symlog")
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
        new_path = self.lang_path.joinpath('language{}_{}.png'.format(agent_index, step))
        plt.savefig(str(new_path))
        plt.close()


class PlotLanguage2Command:
    def __init__(self, lang2_path, params):
        x_left = 0
        x_right = 1.1 if params['stimulus'] == 'quotient' else params['max_num'] + 1
        self.plot_space = linspace(x_left, x_right, 20 * (x_right - x_left), False)
        self.lang2_path = lang2_path

    def __call__(self, agent_index, agent_tuple, step):
        agent = agent_tuple[0]
        lexicon = agent.get_lexicon()

        lang = []

        for word in lexicon:
            category2sth = zip(agent.get_categories(), agent.get_categories_by_word(word))
            fy = [sum([cat.fun(x) * wei for cat, wei in category2sth]) for x in self.plot_space]
            lang.append([word, fy])

        plt.title("language2 in step {} of agent {}".format(step, agent_index))
        # plt.xscale("symlog")
        # plt.yscale("symlog")
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
        new_path = self.lang2_path.joinpath('language{}_{}.png'.format(agent_index, step))
        plt.savefig(str(new_path))
        # if not self.pickle_mode else self.step - self.pickle_step + step + 1))
        plt.close()


class PlotMatrixCommand:

    def __init__(self, matrices_path, params):
        self.matrices_path = matrices_path

    def __call__(self, agent_index, agent_tuple, step):
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
        for i in range(len(lexicon)):
            for j in range(n_categories):
                if lxc_ex[i, j] > 0.0:
                    text = ax.text(j, i, round(lxc_ex[i, j], 2), ha="center", va="center", color="r")
        # for i in range(n_rows):
        #     ax.text(n_cols, i, round(lxc_ex[i, n_cols], 2), ha="center", va="center", color="w")

        ax.set_title("Association matrix")
        fig.tight_layout()
        new_path = self.matrices_path.joinpath('matrix{}_{}.png'.format(agent_index, step))
        plt.savefig(str(new_path))
        plt.close()


class CommandExecutor:
    def __init__(self):
        self.commands = []

    def add_command(self, command):
        self.commands.append(command)

    def execute_commands_in_parallel(self, data_paths, last_population, parallelism=0):
        def chunks(list, chunk_size):
            for i in range(0, len(list), chunk_size):
                yield list[i:i + chunk_size]

        def new_chunked_task(execute_commands, chunk, last_population):
            return Task(execute_commands, chunk=chunk, last_population=last_population)

        # for 0 parallelism is unbounded, we require that len(population) * chunk_size == 200
        if parallelism == 0:
            chunk_size = max(200 / len(last_population), 1)
        else:
            chunk_size = max(len(data_paths) / parallelism, 1)

        tasks = []
        for data_path_chunk in chunks(data_paths, chunk_size):
            tasks.append(new_chunked_task(self.execute_commands, data_path_chunk, last_population))
            tasks[-1].start()

        for task in tasks:
            task.join()

    def execute_commands(self, data_paths, last_population):
        def execute_commands_per_agent(self, agent_index, agent_tuple, step):
            for command_exec in self.commands:
                if agent_tuple[0].language.lxc.size():
                    command_exec(agent_index, agent_tuple, step)

        for path in data_paths:
            step, population = pickle.load(path.open('rb'))
            for agent_index, agent_tuple in enumerate(zip(population, last_population)):
                # assert that zip between agent at current and last step is valid
                assert agent_tuple[0].id == agent_tuple[1].id
                execute_commands_per_agent(self, agent_index, agent_tuple, step)


class Task(Process):
    def __init__(self, execute_commands, chunk, last_population):
        super(Task, self).__init__()
        self.chunk = chunk
        self.last_population = last_population
        self.execute_commands = execute_commands

    def run(self):
        self.execute_commands(self.chunk, self.last_population)


class PlotMonotonicityCommand:

    def __init__(self, root_path):
        self.root_path = root_path
        params = pickle.load(PathProvider.new_path_provider(root_path.joinpath('run0')).get_simulation_params_path().open('rb'))
        StimulusFactory.init(params['stimulus'], params['max_num'])

        self.mon_plot_path = self.root_path.joinpath('stats/mon.pdf')
        self.steps = params['steps']
        self.runs = params['runs']
        self.array = zeros((self.runs, self.steps))

    def fill_array(self):
        logging.debug("Root path %s" % self.root_path)
        for run_num, run_path in enumerate(self.root_path.glob('*')):
            logging.debug("Processing %s, %s" % (run_num, run_path))
            for step_path in PathProvider(run_path).get_data_paths():
                logging.debug("Processing %s" % step_path)
                step, population = pickle.load(step_path.open('rb'))
                # print('run number, step: {}, {}'.format(run_num, step))
                self.array[run_num, step] = population.get_mon()
                logging.debug("mon val %f" % self.array[run_num, step])

    def __call__(self):
        self.fill_array()
        x = range(1, self.steps + 1)
        plt.ylim(bottom=0)
        plt.ylim(top=100)
        plt.xlabel("step")
        plt.ylabel("monotonicity")

        for r in range(self.runs):
            plt.plot(x, [y * 100.0 for y in self.array[r]], '-')

        # plt.legend(['mon'], loc='best')
        plt.savefig(str(self.mon_plot_path))
        plt.close()


class PlotSuccessCommand:

    def __init__(self, root_path):
        self.root_path = root_path
        self.params = pickle.load(PathProvider.new_path_provider(root_path.joinpath('run0')).get_simulation_params_path().open('rb'))
        self.succ_plot_path = self.root_path.joinpath('stats/succ.pdf')
        self.samples_cs1 = []
        self.samples_ds = []
        self.samples_cs2 = []
        self.samples_cs12 = []
        self.cs1_means = []
        self.cs12_means = []
        self.ds_means = []
        self.cs2_means = []
        self.cs_cis_l = []
        self.cs_cis_u = []
        self.cs2_cis_l = []
        self.cs2_cis_u = []
        self.ds_cis_u = []
        self.ds_cis_l = []
        self.cs12_cis_l = []
        self.cs12_cis_u = []

    def get_data(self):
        logging.debug("Root path %s" % self.root_path)
        populations = {}
        for run_num in range(self.params['runs']):
            run_path = self.root_path.joinpath('run' + str(run_num))
            logging.debug("Processing %s, %s" % (run_num, run_path))
            last_step_path = run_path.joinpath('data/step' + str(self.params['steps'] - 1) + '.p')
            step, population = pickle.load(last_step_path.open('rb'))
            populations[run_num] = population
        for step in range(self.params['steps']):
            logging.debug(step)
            self.samples_ds.append([populations[run].ds[step] for run in range(self.params['runs'])])
            self.samples_cs1.append([populations[run].cs1[step] for run in range(self.params['runs'])])
            self.samples_cs2.append([populations[run].cs2[step] for run in range(self.params['runs'])])
            self.samples_cs12.append([populations[run].cs12[step] for run in range(self.params['runs'])])

    def compute_stats(self):
        self.cs1_means = means(self.samples_cs1)
        self.cs2_means = means(self.samples_cs2)
        self.cs12_means = means(self.samples_cs12)
        self.ds_means = means(self.samples_ds)

        self.cs1_cis_l, self.cs1_cis_u = confidence_intervals(self.samples_cs1)
        self.cs2_cis_l, self.cs2_cis_u = confidence_intervals(self.samples_cs2)
        self.ds_cis_l, self.ds_cis_u = confidence_intervals(self.samples_ds)
        self.cs12_cis_l, self.cs12_cis_u = confidence_intervals(self.samples_cs12)

    def plot(self):
        x = range(self.params['steps'])
        plt.ylim(bottom=0)
        plt.ylim(top=100)
        plt.xlabel("step")
        plt.ylabel("success")
        x_ex = range(0, self.params['steps'] + 3)
        th = [self.params['discriminative_threshold'] * 100 for i in x_ex]
        plt.plot(x_ex, th, ':', linewidth=0.2)

        # for r in range(self.params['runs']):
        #    plt.plot(x, self.array_ds[r], 'r--', linewidth=0.5)
        #    plt.plot(x, self.array_cs[r], 'b-', linewidth=0.5)

        plt.plot(x, self.ds_means, 'r-', linewidth=0.3)
        plt.fill_between(x, self.ds_cis_l, self.ds_cis_u,
                         color='r', alpha=.2)

        plt.plot(x, self.cs1_means, 'g--', linewidth=0.3)
        plt.fill_between(x, self.cs1_cis_l, self.cs1_cis_u,
                         color='g', alpha=.2)

        plt.plot(x, self.cs2_means, 'b--', linewidth=0.3)
        plt.fill_between(x, self.cs2_cis_l, self.cs2_cis_u,
                         color='b', alpha=.2)

        plt.plot(x, self.cs12_means, 'k--', linewidth=0.3)
        plt.fill_between(x, self.cs12_cis_l, self.cs12_cis_u,
                         color='k', alpha=.2)

        plt.legend(['dt', 'ds', 'cs1', 'cs2', 'cs12'], loc='best')
        plt.savefig(str(self.succ_plot_path))
        plt.close()

    def __call__(self):
        self.get_data()
        self.compute_stats()
        self.plot()


class PlotNumberOfDSCommand:

    def __init__(self, root_path, threshold=0, active_only=False):
        self.root_path = root_path
        self.threshold = threshold
        self.active_only = active_only

    def get_whole_lexicon(self, run_path, num_agent):
        self.params = pickle.load(
            PathProvider(run_path).get_simulation_params_path().open('rb'))
        self.whole_lexicon = set()
        StimulusFactory.init(self.params['stimulus'], self.params['max_num'])
        for step_path in PathProvider(run_path).get_data_paths():
            _, population = pickle.load(step_path.open('rb'))
            if self.active_only:
                self.whole_lexicon = self.whole_lexicon.union(population.agents[num_agent].get_active_lexicon())
            else:
                self.whole_lexicon=self.whole_lexicon.union(population.agents[num_agent].get_lexicon())
        self.whole_lexicon = list(self.whole_lexicon) # cast to list to preserve the order


    def fill_steps(self, run_path, num_agent):
        for step_path in PathProvider(run_path).get_data_paths():
            current_step, population = pickle.load(step_path.open('rb'))
            for word in self.whole_lexicon:
                i = self.whole_lexicon.index(word)
                try:
                    self.dcnum[i, current_step] = sum(population.agents[num_agent].get_categories_by_word(word) > 0)
                except ValueError:
                    pass

    def plot(self, run_path, num_agent, min_DS_Num):
        f = plt.figure()
        ax = f.add_subplot(111)
        t = arange(self.dcnum.shape[1])
        short_legend = []
        for word in self.whole_lexicon:
            i = self.whole_lexicon.index(word)
            if self.dcnum[i].max() >= min_DS_Num:
                ax.step(t, self.dcnum[i])
                short_legend.append(word)
        ax.legend(short_legend, bbox_to_anchor=(1.04, 1), loc='upper left')
        ax.set_title('Number of Discriminative Categories pinned to the word forms. \n '
                     'Only words having at least '+str(min_DS_Num)+' DC\'s are shown')
        ax.set_xlabel('Step of the simulation')
        ax.set_ylabel('Number of Disciminative Categories')
        if self.active_only:
            out_filename = str(run_path.joinpath('num_of_DC_agent_ACTIVE_'+str(num_agent)+'.png'))
        else:
            out_filename = str(run_path.joinpath('num_of_DC_agent_'+str(num_agent)+'.png'))
        f.savefig(out_filename, bbox_inches="tight")

    def __call__(self, run_num=0, num_agent=0):
        self.get_whole_lexicon(self.root_path.joinpath('run' + str(run_num)), num_agent)
        self.dcnum = zeros([len(self.whole_lexicon), self.params['steps']])

        self.fill_steps(self.root_path.joinpath('run' + str(run_num)), num_agent)
        self.plot(self.root_path.joinpath('run' + str(run_num)), num_agent, 2)

        print self.dcnum

        print self.whole_lexicon
        # for run_num, run_path in enumerate(self.root_path.glob('*')):
        #     for step in  PathProvider(run_path).get_data_paths():
        #         print '{}  {}'.format(run_path, step)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    parser = argparse.ArgumentParser(prog='plotting data')

    parser.add_argument('--data_root', '-d', help='root path to {data, cats, langs, matrices, ...}', type=str,
                        default="simulation")
    parser.add_argument('--plot_cats', '-c', help='plot categories', type=bool, default=False)
    parser.add_argument('--plot_langs', '-l', help='plot languages', type=bool, default=False)
    parser.add_argument('--plot_langs2', '-l2', help='plot languages 2', type=bool, default=False)
    parser.add_argument('--plot_matrices', '-m', help='plot matrices', type=bool, default=False)
    parser.add_argument('--plot_success', '-s', help='plot success', type=bool, default=False)
    parser.add_argument('--plot_mon', '-mon', help='plot success', type=bool, default=False)
    parser.add_argument('--plot_num_DS', '-nds', help='plot success', type=bool, default=False)
    parser.add_argument('--parallelism', '-p', help='number of processes (unbounded if 0)', type=int, default=8)

    parsed_params = vars(parser.parse_args())

    logging.debug("loading pickled simulation from '%s' file", parsed_params['data_root'])
    data_root_path = Path(parsed_params['data_root'])

    if not data_root_path.exists():
        logging.debug("Path %s does not exist" % data_root_path.absolute())
        exit()

    if parsed_params['plot_mon']:
        plot_mon_command = PlotMonotonicityCommand(Path(parsed_params['data_root']))
        plot_mon_command()

    if parsed_params['plot_num_DS']:
        plot_num_DS_command = PlotNumberOfDSCommand(Path(parsed_params['data_root']), active_only=True)
        plot_num_DS_command()

    if parsed_params['plot_success']:
        logging.debug('start plot success')
        plot_success_command = PlotSuccessCommand(Path(parsed_params['data_root']))
        plot_success_command()

    # set commands to be executed
    for data_path in Path(parsed_params['data_root']).glob('run[0-9]*'):
        path_provider = PathProvider.new_path_provider(data_path)
        command_executor = CommandExecutor()
        params = pickle.load(path_provider.get_simulation_params_path().open('rb'))

        if parsed_params['plot_cats']:
            command_executor.add_command(PlotCategoryCommand(path_provider.cats_path, params))
        if parsed_params['plot_langs']:
            command_executor.add_command(PlotLanguageCommand(path_provider.lang_path, params))
        if parsed_params['plot_langs2']:
            command_executor.add_command(PlotLanguage2Command(path_provider.lang2_path, params))
        if parsed_params['plot_matrices']:
            command_executor.add_command(PlotMatrixCommand(path_provider.matrices_path, params))

        path_provider.create_directories()

        last_step = params['steps'] - 1
        _, last_population = pickle.load(path_provider.get_simulation_step_path(last_step).open('rb'))
        path_provider.get_simulation_step_path(last_step)

        start_time = time.time()
        # PlotMonotonicity(parsed_params['data_root'])()
        data_paths = path_provider.get_data_paths()
        if parsed_params['parallelism'] == 1:
            command_executor.execute_commands(data_paths, last_population)
        else:
            command_executor.execute_commands_in_parallel(data_paths, last_population, parsed_params['parallelism'])

        logging.debug('execution time {}sec, with params {}'.format(time.time() - start_time, parsed_params))
