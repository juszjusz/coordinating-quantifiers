import matplotlib
from pathlib import Path

from inmemory_calculus import inmem
from path_provider import PathProvider
from plot_agent import PlotCategoryCommand, PlotLanguageCommand, PlotLanguage2Command, PlotMatrixCommand, \
    CommandExecutor
from stats import confidence_intervals, means

matplotlib.use('Agg')

import argparse
import logging
import pickle
import sys
import time

import matplotlib.pyplot as plt
from numpy import arange, zeros, asarray
import h5py


class MakeHdf5:
    def __init__(self, root_path, stimuluses, params):
        self.root_path1 = Path(root_path)
        self.stimuluses = stimuluses
        self.params = params

        # logging.critical(self.steps)
        # self.steps = range(3, self.params['steps'])
        self.steps = [max(step * 10 - 1, 0) for step in range(1 + int(self.params['steps'] / 10))]
        # self.hdf_path = self.root_path1.joinpath('stats/meanings.h5')

    def __call__(self):
        f = h5py.File("m.h5", "w")
        for run_num, run_path in enumerate(self.root_path1.glob('run[0-9]*')):
            hdf_list = []
            for step in self.steps:
                logging.debug("Run %d, step %d" % (run_num, step))
                step, population = pickle.load(run_path.joinpath("data/step" + str(step) + ".p").open('rb'))
                meanings = population.get_meanings(unpickled_stimuluses)
                hdf_list.append(meanings)
            hdf_arr = asarray(hdf_list)
            f.create_dataset(name='run%d' % run_num, data=hdf_arr)
        f.close()
        logging.debug("FINISHED")


class PlotConvexityCommand:

    def __init__(self, root_paths, stimuluses, params):
        self.root_path2 = None
        self.root_path1 = Path(root_paths[0])
        if len(root_paths) > 1:
            self.root_path2 = Path(root_paths[1])

        self.stimuluses = stimuluses
        self.params = params

        self.steps = [max(step * 10 - 1, 0) for step in range(1 + self.params['steps'] / 10)]
        logging.critical(self.steps)
        # self.steps = range(0, self.params['steps'])
        self.conv_plot_path = self.root_path1.joinpath('stats/convexity.pdf')
        # self.array1 = zeros((self.params['runs'], self.steps))
        self.conv_samples1 = []
        # self.array2 = zeros((self.params['runs'], self.steps))
        self.conv_samples2 = []
        self.conv_means1 = []
        self.conv_cis1_l = []
        self.conv_cis1_u = []
        self.conv_means2 = []
        self.conv_cis2_l = []
        self.conv_cis2_u = []

    def get_data(self):
        # logging.debug("Root path %s" % self.root_path1)
        for step in self.steps:
            logging.debug("Processing step %d" % step)
            sample = []
            for run_num, run_path in enumerate(self.root_path1.glob('run[0-9]*')):
                # logging.debug("Processing %s, %s" % (run_num, run_path))
                # logging.debug("Processing %s" % "step" + str(step) + ".p")
                step, population = pickle.load(run_path.joinpath("data/step" + str(step) + ".p").open('rb'))
                sample.append(population.get_convexity(self.stimuluses))
                # logging.debug("mon val %f" % sample[-1])
            self.conv_samples1.append(sample)
        # for step in range(self.params['steps']):
        #    self.mon_samples1.append(list(self.array1[:, max(step*100-1, 0)]))

        if self.root_path2 is not None:
            logging.debug("Root path %s" % self.root_path2)
            for step in self.steps:
                logging.debug("Processing step %d" % step)
                sample = []
                for run_num, run_path in enumerate(self.root_path2.glob('run[0-9]')):
                    logging.debug("Processing %s, %s" % (run_num, run_path))
                    # for step_path in PathProvider(run_path).get_data_paths():
                    # logging.debug("Processing %s" % step_path)
                    step, population = pickle.load(run_path.joinpath("data/step" + str(step) + ".p").open('rb'))
                    # self.array2[run_num, step] = population.get_mon()
                    # logging.debug("mon val %f" % self.array2[run_num, step])
                    sample.append(population.get_convexity(self.stimuluses))
                self.conv_samples2.append(sample)
                # for step in range(self.params['steps']):
                # self.mon_samples2.append(list(self.array2[:, step]))

    def compute_stats(self):
        logging.debug('in compute_stats')
        self.conv_means1 = means(self.conv_samples1)
        # logging.debug(len(self.mon_means1))

        self.conv_cis1_l, self.conv_cis1_u = confidence_intervals(self.conv_samples1)

        if self.root_path2 is not None:
            self.mon_means2 = means(self.conv_samples2)
            self.mon_cis2_l, self.mon_cis2_u = confidence_intervals(self.conv_samples2)

    def plot(self):
        # x = range(1, self.params['steps'] + 1)
        x = self.steps
        plt.ylim(bottom=50)
        plt.ylim(top=105)
        plt.xlabel("step")
        plt.ylabel("convexity")

        # for r in range(self.runs):
        #    plt.plot(x, [y * 100.0 for y in self.array[r]], '-')

        plt.plot(x, self.conv_means1, 'k-', linewidth=0.3)

        for i in range(0, self.params['runs']):
            logging.critical("run %d" % i)
            plt.plot(x, [self.conv_samples1[s][i] for s in range(0, len(self.steps))], 'k-', linewidth=0.2, alpha=.3)

        if self.root_path2 is not None:
            plt.plot(x, self.conv_means2, 'b--', linewidth=0.3)
            plt.fill_between(x, self.conv_cis2_l, self.conv_cis2_u,
                             color='b', alpha=.2)
            plt.legend([str(self.root_path1), str(self.root_path2)], loc='best')
        else:
            # plt.legend([str(self.root_path1)], loc='lower right')
            plt.legend(['convexity'], loc='lower right')
        plt.savefig(str(self.conv_plot_path))
        plt.close()

    def __call__(self):
        self.get_data()
        self.compute_stats()
        self.plot()


class PlotMonotonicityCommand:

    def __init__(self, root_paths, stimuluses, params):
        self.root_path2 = None
        self.root_path1 = Path(root_paths[0])
        if len(root_paths) > 1:
            self.root_path2 = Path(root_paths[1])

        self.stimuluses = stimuluses
        self.params = params
        self.steps = [max(step * 100 - 1, 0) for step in range(1 + int(self.params['steps'] / 100))]
        self.mon_plot_path = Path('.').joinpath('monotonicity.pdf')
        # self.array1 = zeros((self.params['runs'], self.steps))
        self.mon_samples1 = []
        # self.array2 = zeros((self.params['runs'], self.steps))
        self.mon_samples2 = []
        self.mon_means1 = []
        self.mon_cis1_l = []
        self.mon_cis1_u = []
        self.mon_means2 = []
        self.mon_cis2_l = []
        self.mon_cis2_u = []

    def get_data(self):
        # logging.debug("Root path %s" % self.root_path1)
        for step in self.steps:
            # logging.debug("Processing step %d" % step)
            sample = []
            for run_num, run_path in enumerate(self.root_path1.glob('run[0-9]*')):
                # logging.debug("Processing %s, %s" % (run_num, run_path))
                # logging.debug("Processing %s" % "step" + str(step) + ".p")
                step, population = pickle.load(run_path.joinpath("data/step" + str(step) + ".p").open('rb'))
                sample.append(population.get_mon(self.stimuluses))
                # logging.debug("mon val %f" % sample[-1])
            self.mon_samples1.append(sample)
        # for step in range(self.params['steps']):
        #    self.mon_samples1.append(list(self.array1[:, max(step*100-1, 0)]))

        if self.root_path2 is not None:
            logging.debug("Root path %s" % self.root_path2)
            for step in self.steps:
                logging.debug("Processing step %d" % step)
                sample = []
                for run_num, run_path in enumerate(self.root_path2.glob('run[0-9]')):
                    logging.debug("Processing %s, %s" % (run_num, run_path))
                    # for step_path in PathProvider(run_path).get_data_paths():
                    # logging.debug("Processing %s" % step_path)
                    step, population = pickle.load(run_path.joinpath("data/step" + str(step) + ".p").open('rb'))
                    # self.array2[run_num, step] = population.get_mon()
                    # logging.debug("mon val %f" % self.array2[run_num, step])
                    sample.append(population.get_mon(self.stimuluses))
                self.mon_samples2.append(sample)
                # for step in range(self.params['steps']):
                # self.mon_samples2.append(list(self.array2[:, step]))

    def compute_stats(self):
        logging.debug('in compute_stats')
        self.mon_means1 = means(self.mon_samples1)
        # logging.debug(len(self.mon_means1))

        self.mon_cis1_l, self.mon_cis1_u = confidence_intervals(self.mon_samples1)

        if self.root_path2 is not None:
            self.mon_means2 = means(self.mon_samples2)
            self.mon_cis2_l, self.mon_cis2_u = confidence_intervals(self.mon_samples2)

    def plot(self):
        # x = range(1, self.params['steps'] + 1)
        x = self.steps
        plt.ylim(bottom=0)
        plt.ylim(top=100)
        plt.xlabel("step")
        plt.ylabel("monotonicity")

        # for r in range(self.runs):
        #    plt.plot(x, [y * 100.0 for y in self.array[r]], '-')

        plt.plot(x, self.mon_means1, 'r--', linewidth=0.3)
        plt.fill_between(x, self.mon_cis1_l, self.mon_cis1_u,
                         color='r', alpha=.2)

        if self.root_path2 is not None:
            plt.plot(x, self.mon_means2, 'b--', linewidth=0.3)
            plt.fill_between(x, self.mon_cis2_l, self.mon_cis2_u,
                             color='b', alpha=.2)
            plt.legend(['mon. no ANS', 'mon. ANS'], loc='best')
        else:
            plt.legend([str(self.root_path1)], loc='best')

        plt.savefig(str(self.mon_plot_path))
        plt.close()

    def __call__(self):
        self.get_data()
        self.compute_stats()
        self.plot()

    # @staticmethod
    # def plot_means(str_data_roots):
    #     stats_dict = PlotMonotonicityCommand.prepare_stats(str_data_roots)
    #     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    #     steps = len(stats_dict.values()[0][0])
    #     logging.debug("steps %d" % steps)
    #
    #     x = range(1, steps + 1)
    #     plt.ylim(bottom=0)
    #     plt.ylim(top=100)
    #     plt.xlabel("step")
    #     plt.ylabel("monotonicity")
    #
    #     for name in str_data_roots:
    #         stats = stats_dict[name]
    #         color = colors[str_data_roots.index(name)]
    #         plt.plot(x, stats[0], color, linewidth=0.3)
    #         logging.debug(stats[1])
    #         logging.debug(stats[2])
    #         plt.fill_between(x, stats[1], stats[2], color, alpha=.2)
    #
    #         #plt.legend([sorted_str_data_roots[i]], loc='best')
    #         #plt.legend(['mon'], loc='best')
    #
    #     plt.legend(str_data_roots, loc='best')
    #     plt.savefig(str(Path('./monotonicty.pdf')))
    #     plt.close()


class PlotSuccessCommand:

    def __init__(self, root_path, stimuluses, params):
        self.root_path = root_path
        self.params = params
        self.stimuluses = stimuluses
        self.succ_plot_path = self.root_path.joinpath('stats/succ.pdf')
        self.steps = [max(step * 10 - 1, 0) for step in range(1 + int(self.params['steps'] / 10))]
        self.samples_cs1 = []
        self.samples_ds = []
        self.samples_cs2 = []
        self.samples_cs12 = []
        self.samples_nw = []
        self.nw_means = []
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
        self.nw_cis_l = []
        self.nw_cis_u = []

    def prepare_data(self):
        logging.debug("Root path %s" % self.root_path)
        psize = self.params['population_size']
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
        for step in self.steps:
            logging.debug(step)
            nw_sample = []
            for r in range(self.params['runs']):
                run_path = self.root_path.joinpath('run' + str(r))
                rsp = run_path.joinpath('data/step' + str(step) + '.p')
                step, population = pickle.load(rsp.open('rb'))
                nw_sample.append(sum([len(a.get_active_lexicon(self.stimuluses)) for a in population.agents]) / psize)
            self.samples_nw.append(nw_sample)

    def compute_stats(self):
        self.cs1_means = means(self.samples_cs1)
        self.cs2_means = means(self.samples_cs2)
        self.cs12_means = means(self.samples_cs12)
        self.ds_means = means(self.samples_ds)
        self.nw_means = means(self.samples_nw)
        # logging.debug(self.nw_means)

        self.nw_cis_l, self.nw_cis_u = confidence_intervals(self.samples_nw)
        self.cs1_cis_l, self.cs1_cis_u = confidence_intervals(self.samples_cs1)
        self.cs2_cis_l, self.cs2_cis_u = confidence_intervals(self.samples_cs2)
        self.ds_cis_l, self.ds_cis_u = confidence_intervals(self.samples_ds)
        self.cs12_cis_l, self.cs12_cis_u = confidence_intervals(self.samples_cs12)

    def plot(self):
        x = range(self.params['steps'])
        x100 = self.steps
        fig, ax1 = plt.subplots()
        plt.ylim(bottom=0)
        plt.ylim(top=100)
        plt.xlabel("step")
        plt.ylabel("success")
        # x_ex = range(0, self.params['steps'] + 3)
        # th = [self.params['discriminative_threshold'] * 100 for i in x_ex]
        # plt.plot(x_ex, th, ':', linewidth=0.2)

        # for r in range(self.params['runs']):
        #    plt.plot(x, self.array_ds[r], 'r--', linewidth=0.5)
        #    plt.plot(x, self.array_cs[r], 'b-', linewidth=0.5)

        plt.plot(x, self.ds_means, 'r-', linewidth=0.6)
        for i in range(0, self.params['runs']):
            plt.plot(x, [self.samples_ds[s][i] for s in range(0, self.params['steps'])], 'r-', linewidth=0.2, alpha=.3)
        # plt.fill_between(x, self.ds_cis_l, self.ds_cis_u,
        #                 color='r', alpha=.2)

        plt.plot(x, self.cs1_means, 'g--', linewidth=0.6)
        for i in range(0, self.params['runs']):
            plt.plot(x, [self.samples_cs1[s][i] for s in range(0, self.params['steps'])], 'g-', linewidth=0.2, alpha=.3)
        # plt.fill_between(x, self.cs1_cis_l, self.cs1_cis_u,
        #                 color='g', alpha=.2)

        # ax1.legend(['discrimination', 'communication'], loc='lower right')
        ax2 = ax1.twinx()
        ax2.set_ylabel('|active lexicon|')
        # ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.plot(x100, self.nw_means, 'b--', linewidth=0.3)
        # for i in range(0, self.params['runs']):
        #    ax2.plot(x100, [self.samples_nw[s][i] for s in range(0, len(self.steps))], 'b-', linewidth=0.2, alpha=.3)
        ax2.fill_between(x100, self.nw_cis_l, self.nw_cis_u,
                         color='b', alpha=.2)
        ax2.set_yticks(range(0, 15, 1),
                       ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'))
        # ,'10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29'
        ax2.tick_params(axis='y')
        # ax2.legend(['size'], loc='upper right')

        # plt.plot(x, self.cs2_means, 'b--', linewidth=0.3)
        # plt.fill_between(x, self.cs2_cis_l, self.cs2_cis_u,
        #                color='b', alpha=.2)

        # plt.plot(x, self.cs12_means, 'k--', linewidth=0.3)
        # plt.fill_between(x, self.cs12_cis_l, self.cs12_cis_u,
        #                 color='k', alpha=.2)

        # plt.legend(['dt', 'ds', 'cs1', 'cs2', 'cs12'], loc='best')

        fig.tight_layout()
        plt.savefig(str(self.succ_plot_path))
        plt.close()

    def __call__(self):
        self.prepare_data()
        self.compute_stats()
        self.plot()


class PlotNumberOfDSCommand:

    def __init__(self, root_path, stimuluses, params, threshold=0, active_only=False):
        self.root_path = root_path
        self.stimuluses = stimuluses
        self.params = params
        self.threshold = threshold
        self.active_only = active_only

    def get_whole_lexicon(self, run_path, num_agent):
        self.whole_lexicon = set()
        for step_path in PathProvider(run_path).get_data_paths():
            _, population = pickle.load(step_path.open('rb'))
            if self.active_only:
                self.whole_lexicon = self.whole_lexicon.union(
                    population.agents[num_agent].get_active_lexicon(self.stimuluses))
            else:
                self.whole_lexicon = self.whole_lexicon.union(population.agents[num_agent].get_lexicon())
        self.whole_lexicon = list(self.whole_lexicon)  # cast to list to preserve the order

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
                     'Only words having at least ' + str(min_DS_Num) + ' DC\'s are shown')
        ax.set_xlabel('Step of the simulation')
        ax.set_ylabel('Number of Disciminative Categories')
        if self.active_only:
            out_filename = str(run_path.joinpath('num_of_DC_agent_ACTIVE_' + str(num_agent) + '.png'))
        else:
            out_filename = str(run_path.joinpath('num_of_DC_agent_' + str(num_agent) + '.png'))
        f.savefig(out_filename, bbox_inches="tight")

    def __call__(self, run_num=0, num_agent=0):
        self.get_whole_lexicon(self.root_path.joinpath('run' + str(run_num)), num_agent)
        self.dcnum = zeros([len(self.whole_lexicon), self.params['steps']])

        self.fill_steps(self.root_path.joinpath('run' + str(run_num)), num_agent)
        self.plot(self.root_path.joinpath('run' + str(run_num)), num_agent, 2)

        print(self.dcnum)

        print(self.whole_lexicon)
        # for run_num, run_path in enumerate(self.root_path.glob('*')):
        #     for step in  PathProvider(run_path).get_data_paths():
        #         print '{}  {}'.format(run_path, step)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    parser = argparse.ArgumentParser(prog='plotting data')

    parser.add_argument('--data_root', '-d', help='root path to {data, cats, langs, matrices, ...}', type=str,
                        default="test")
    parser.add_argument('--plot_cats', '-c', help='plot categories', type=bool, default=True)
    parser.add_argument('--plot_langs', '-l', help='plot languages', type=bool, default=True)
    parser.add_argument('--plot_langs2', '-l2', help='plot languages 2', type=bool, default=True)
    parser.add_argument('--plot_matrices', '-m', help='plot matrices', type=bool, default=False)
    parser.add_argument('--plot_success', '-s', help='plot success', type=bool, default=False)
    parser.add_argument('--plot_mon', '-mon', help='plot monotonicity', type=bool, default=False)
    parser.add_argument('--plot_mons', '-mons', help='plot monotonicity', type=str, nargs='+', default='')
    parser.add_argument('--plot_conv', '-conv', help='plot convexity', type=str, nargs='+', default='')
    parser.add_argument('--plot_num_DS', '-nds', help='plot success', type=bool, default=False)
    parser.add_argument('--hdf_franek', '-hdf', help='hdf franek', type=bool, default=False)
    parser.add_argument('--parallelism', '-p', help='number of processes (unbounded if 0)', type=int, default=8)

    parsed_params = vars(parser.parse_args())

    logging.debug("loading pickled simulation from '%s' file", parsed_params['data_root'])
    data_root_path = Path(parsed_params['data_root'])
    sim_params = pickle.load(PathProvider.new_path_provider(data_root_path).get_simulation_params_path().open('rb'))
    unpickled_inmem = pickle.load(PathProvider.new_path_provider(data_root_path).get_inmem_calc_path().open('rb'))
    unpickled_stimuluses = pickle.load(PathProvider.new_path_provider(data_root_path).get_stimuluses_path().open('rb'))

    for k, v in unpickled_inmem.items():
        inmem[k] = v

    if parsed_params['hdf_franek']:
        hdf_command = MakeHdf5(parsed_params['data_root'], unpickled_stimuluses, sim_params)
        hdf_command()

    if len(parsed_params['plot_mons']) == 2:
        logging.critical("PLOTTING MONOTONICITY")
        data_root_path1 = Path(parsed_params['plot_mons'][0])
        unpickled_stimuluses1 = pickle.load(
            PathProvider.new_path_provider(data_root_path1).get_stimuluses_path().open('rb'))
        simulation1_params = pickle.load(
            PathProvider.new_path_provider(data_root_path1).get_simulation_params_path().open('rb'))
        pmc = PlotMonotonicityCommand(parsed_params['plot_mons'], unpickled_stimuluses1, simulation1_params)
        pmc()

    if not data_root_path.exists():
        logging.debug("Path %s does not exist" % data_root_path.absolute())
        exit()

    if parsed_params['plot_mon']:
        plot_mon_command = PlotMonotonicityCommand([parsed_params['data_root']], unpickled_stimuluses, sim_params)
        plot_mon_command()

    if parsed_params['plot_conv']:
        plot_conv_command = PlotConvexityCommand([parsed_params['data_root']], unpickled_stimuluses, sim_params)
        plot_conv_command()

    if parsed_params['plot_num_DS']:
        plot_num_DS_command = PlotNumberOfDSCommand(Path(parsed_params['data_root']), unpickled_stimuluses, sim_params,
                                                    active_only=True)
        plot_num_DS_command()

    if parsed_params['plot_success']:
        logging.debug('start plot success')
        plot_success_command = PlotSuccessCommand(Path(parsed_params['data_root']), unpickled_stimuluses, sim_params)
        plot_success_command()

    # set commands to be executed
    for data_path in Path(parsed_params['data_root']).glob('run[0-9]*'):
        path_provider = PathProvider.new_path_provider(data_path)
        command_executor = CommandExecutor()

        if parsed_params['plot_cats']:
            command_executor.add_command(PlotCategoryCommand(inmem))
        if parsed_params['plot_langs']:
            command_executor.add_command(PlotLanguageCommand(inmem))
        if parsed_params['plot_langs2']:
            command_executor.add_command(PlotLanguage2Command(inmem))
        if parsed_params['plot_matrices']:
            command_executor.add_command(PlotMatrixCommand())

        start_time = time.time()
        # PlotMonotonicity(parsed_params['data_root'])()
        data_paths = path_provider.get_data_paths()

        command_executor.execute_commands(data_paths, data_path, sim_params['steps'] - 1, parsed_params['parallelism'])

        logging.log(logging.INFO,
                    'execution time {}sec, with params {}'.format(time.time() - start_time, parsed_params))
