import itertools
import logging
import pathlib
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import seaborn as sns

from matplotlib import animation
from matplotlib import pyplot as plt

from path_provider import PathProvider

_default_ext = 'gif'

writers = {
    'gif': animation.PillowWriter(fps=100),
    'html': animation.HTMLWriter(fps=100),
}


class PlotLanguageCommand:
    def __init__(self, inmem, ext=_default_ext):
        self.inmem = inmem
        self.domain = self.inmem['DOMAIN']
        self.ext = ext

    def __call__(self, agent, target_path):
        fig = plt.figure()

        categories_in_steps = [a.get_categories() for a in agent]
        max_peek = max(c.discretized_distribution(self.inmem["REACTIVE_UNIT_DIST"]).max() for categories in
                       categories_in_steps for c in categories)
        cat_size = max(len(c) for c in categories_in_steps)

        ax = plt.axes(xlim=(self.domain.min(), self.domain.max()), ylim=(0, max_peek + 1000))
        plt.yscale('symlog', linthresh=100)
        lines = [ax.plot([], [], lw=3)[0] for _ in range(0, cat_size)]

        step_label = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def init():
            step_label.set_text('step 0')

            for line in lines:
                line.set_data([], [])

            return lines

        def animate(step):
            step_label.set_text('language in step {} of agent {}'.format(step, agent[0].id))

            agent_at_step = agent[step]
            categories = agent_at_step.get_categories()

            for category_index, category in enumerate(categories):
                word_sorted_by_val = agent_at_step.language.get_words_sorted_by_val(category_index, threshold=0)
                if word_sorted_by_val:
                    # pick a word with a highest connectivity with a given category
                    word = word_sorted_by_val[0]

                    x = self.inmem["DOMAIN"]
                    y = category.discretized_distribution(self.inmem["REACTIVE_UNIT_DIST"])

                    lines[category_index].set_data(x, y)
                    lines[category_index].set_label(word)
                    plt.legend()

            return lines

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(agent), interval=200, blit=True)

        path = target_path.joinpath('agent-{}-language.{}'.format(agent[0].id, self.ext))
        anim.save(path, writer=writers[self.ext])


class PlotLanguage2Command:
    def __init__(self, inmem, ext=_default_ext):
        self.inmem = inmem
        self.domain = self.inmem['DOMAIN']
        self.ext = ext

    def __call__(self, agent, target_path):
        fig = plt.figure()
        ax = plt.axes(xlim=(self.domain.min(), self.domain.max()), ylim=(0, 2000))
        max_lexicon = max(len(step.get_lexicon()) for step in agent)
        lines = [ax.plot([], [], lw=3)[0] for _ in range(0, max_lexicon)]

        plt.yscale('symlog', linthresh=100)

        step_label = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def init():
            step_label.set_text('step 0')

            for line in lines:
                line.set_data([], [])

            return lines

        def animate(step):
            step_label.set_text("language2 in step {} of agent {}".format(step, agent[0].id))

            agent_at_step = agent[step]

            for i, word in enumerate(agent_at_step.get_lexicon()):
                category_to_weights = zip(agent_at_step.get_categories(), agent_at_step.get_categories_by_word(word))

                y = sum(cat.union(self.inmem['REACTIVE_UNIT_DIST']) * wei for cat, wei in category_to_weights)

                lines[i].set_data(self.domain, y)
                lines[i].set_label(word)
                plt.legend()

            return lines

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(agent), interval=1, blit=True)

        path = target_path.joinpath('agent-{}-language2.{}'.format(agent[0].id, self.ext))
        anim.save(path, writer=writers[self.ext])


class PlotCategoryCommand:
    def __init__(self, inmem, ext=_default_ext):
        self.inmem = inmem
        self.domain = self.inmem['DOMAIN']
        self.ext = ext

    def __call__(self, agent, target_path):
        fig = plt.figure()

        categories_in_steps = [step.get_categories() for step in agent]
        cat_size = max(len(c) for c in categories_in_steps)
        max_peek = max(c.discretized_distribution(self.inmem["REACTIVE_UNIT_DIST"]).max() for categories in
                       categories_in_steps for c in categories)

        ax = plt.axes(xlim=(self.domain.min(), self.domain.max()), ylim=(0, max_peek + 1000))

        plt.yscale('symlog', linthresh=100)

        lines = [ax.plot([], [], lw=3)[0] for _ in range(0, cat_size)]

        step_label = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def init():
            step_label.set_text('step 0')

            for line in lines:
                line.set_data([], [])

            return lines

        def animate(step):
            step_label.set_text('step {}'.format(step))

            cats = agent[step].get_categories()

            for cat, line in zip(cats, lines):
                y = cat.discretized_distribution(self.inmem["REACTIVE_UNIT_DIST"])
                line.set_label(cat.id)
                line.set_data(self.domain, y)
                plt.legend()

            return lines

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(agent), interval=10, blit=True)

        path = target_path.joinpath('agent-{}-category.{}'.format(agent[0].id, self.ext))
        anim.save(path, writer=writers[self.ext])


class PlotMatrixCommand:

    def __init__(self, ext=_default_ext):
        self.log_th = 4
        self.ext = ext

    def __call__(self, agent, target_path):
        lxcs = [step.language.lxc.to_array() for step in agent]

        max_rows, max_cols = lxcs[-1].shape

        lexicon = agent[-1].language.get_full_lexicon()

        max_lxc_connection = max(a.language.lxc.to_array().max(initial=0) for a in agent)

        max_cell_value = np.log(max_lxc_connection) if max_lxc_connection > self.log_th else max_lxc_connection

        grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}

        fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(12, 8))

        def animate(step):
            lxc = lxcs[step]

            rows, cols = lxc.shape

            heat_map0 = np.zeros((max_rows, max_cols))

            heat_map0[:rows, :cols] = lxc

            map_ = heat_map0 > self.log_th

            heat_map0[map_] = np.log(heat_map0[map_])

            sns.heatmap(heat_map0,
                        ax=ax,
                        annot=False,
                        cbar=True,
                        cmap="mako",
                        cbar_ax=cbar_ax,
                        vmin=0,
                        vmax=max_cell_value,
                        yticklabels=lexicon)

        anim = animation.FuncAnimation(fig=fig, func=animate, frames=len(agent), interval=50, blit=False)

        path = target_path.joinpath('agent-{}-language-category-matrix.{}'.format(agent[0].id, self.ext))
        anim.save(path, writer=writers[self.ext])


class CommandExecutor:
    def __init__(self):
        self.commands = []

    def add_command(self, command):
        self.commands.append(command)

    def execute_commands(self, data_paths, data_path, last_step, parallelism):
        agents_in_steps = dict()

        for path in data_paths:
            _, last_population = pickle.load(PathProvider(data_path).get_simulation_step_path(last_step).open('rb'))
            step, population = pickle.load(path.open('rb'))
            for agent in population.agents:
                if not agents_in_steps.get(agent.id):
                    agents_in_steps[agent.id] = [None] * (last_step + 1)
                agent.language.word_gen = None
                agents_in_steps[agent.id][step] = agent

        target_path = pathlib.Path(data_path).absolute()

        with ProcessPoolExecutor(parallelism) as executor:
            for agent, command in itertools.product(agents_in_steps.values(), self.commands):
                logging.info('submitting {} command with agent {}'.format(agent[0].id, command))
                if True or parallelism == 1:
                    if agent[0].id == 1:
                        command(agent, target_path)
                else:
                    executor.submit(command, agent, target_path)
