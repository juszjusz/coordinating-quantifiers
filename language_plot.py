import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from numpy import linspace, array, zeros, column_stack, arange, log, amax


class CategoriesPlot:
    def __init__(self, x_left=0, x_right=100):
        self.plot_space = linspace(x_left, x_right, 20 * (x_right - x_left), False)

    def plot_categories(self, population, params, step):
        for agent_index, agent in enumerate(population):
            self.plot_category(agent_index, agent, step)

    def plot_category(self, agent_index, agent, step):
        plt.title("categories")
        ax = plt.gca()
        plt.xscale("symlog")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        plt.yscale("symlog")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        cats = agent.get_categories()

        linestyles = dict(zip(cats,
                              [(c, s) for s in ['solid', 'dotted', 'dashed', 'dashdot'] for c in
                               sns.color_palette()]))

        for cat in cats:
            graph = [cat.fun(x_0) for x_0 in self.plot_space]
            color = linestyles[cat][0]
            linestyle = linestyles[cat][1]

            plt.plot(self.plot_space, graph,
                     color=color,
                     linestyle=linestyle,
                     label="%d" % (cat.id + 1))

        plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1, 1))
        plt.tight_layout(pad=0)

        plt.savefig("./simulation_results/cats/categories%d_%d" % (
            agent_index, step))
        plt.close()


class LanguagePlot:
    def __init__(self, x_left=0, x_right=100):
        self.plot_space = linspace(x_left, x_right, 20 * (x_right - x_left), False)

    def plot_languages(self, population, params, step):
        for agent_index, agent in enumerate(population):
            word_linestyles = dict(zip(agent.get_lexicon(),
                                       [(c, s) for s in ['solid', 'dotted', 'dashed', 'dashdot'] for c in
                                        sns.color_palette()]))
            language = self.convert_agent(agent)
            self.plot_language(agent_index, language, step, word_linestyles)

    def convert_agent(self, agent):
        lang = []
        lexicon = agent.get_lexicon()
        categories = agent.get_categories()
        forms_to_categories = {form: [] for form in lexicon}

        for (category_index, _) in enumerate(categories):
            words_by_category = agent.language.get_words_by_category(category_index)
            m = max(words_by_category)
            if m > 0:
                max_form_indices = [ind for ind, w in enumerate(words_by_category) if w == m]
                form = lexicon[max_form_indices[0]]
                forms_to_categories[form].append(category_index)

        for word in range(len(lexicon)):
            f = lexicon[word]
            if len(forms_to_categories[f]) == 0:
                continue
            else:
                lang.append([f])
                for j in forms_to_categories[f]:
                    lang[-1].append([categories[j].fun(x_0) for x_0 in self.plot_space])

        return lang

    def plot_language(self, agent_index, lang, step, word_to_linestyles):
        plt.title("language")
        plt.xscale("symlog")
        plt.yscale("symlog")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        for word_cats_index in range(len(lang)):
            word_cats = lang[word_cats_index]
            f = word_cats[0]
            ls = word_to_linestyles[f]
            for y in word_cats[1::]:
                plt.plot(self.plot_space, y, color=ls[0], linestyle=ls[1])
            plt.plot([], [], color=ls[0], linestyle=ls[1], label=f)
        plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1, 1))
        plt.tight_layout(pad=0)
        plt.savefig("./simulation_results/langs/language%d_%d.png" % (agent_index, step))
        plt.close()


class PlotMatrices:

    def store_matrices(self, population):
        for agent in population:
            self.plot_matrix(agent)

    def plot_matrix(self, agent):
        matrix = array(agent.language.lxc.matrix)
        lang = list(agent.get_lexicon())
        cats = [c.id for c in agent.language.categories]

        self._shape_[i] = (max(self._shape_[i][0], lxc.shape[0]), max(self._shape_[i][1], lxc.shape[1]))
        # self.matrices[i].append((list(lex), array(lxc), cats))
        # matrix = (list(lex), array(lxc), cats)
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
