from bokeh.plotting import figure
from pathlib import Path
from typing import List

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from calculator import Calculator, Stimulus
from domain_objects import NewCategory


def plot_categories(agent: str, calculator: Calculator, categories: List[NewCategory], step):
    def new_linestyles(seq):
        linestyles = [(color, style) for style in ['solid', 'dotted', 'dashed', 'dashdot'] for color in
                      sns.color_palette()]
        return dict(zip(seq, linestyles))

    plt.title("categories")
    ax = plt.gca()
    # plt.xscale("symlog")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.yscale("symlog")
    ax.yaxis.set_major_formatter(ScalarFormatter())

    linestyles = new_linestyles(categories)

    for cat in categories:
        color, linestyle = linestyles[cat]

        plt.plot(calculator.domain(), cat.discretized_distribution(calculator),
                 color=color,
                 linestyle=linestyle,
                 label="%d" % (cat.category_id))

    plt.legend(loc='upper left', prop={'size': 6}, bbox_to_anchor=(1, 1))
    plt.tight_layout(pad=0)
    new_path = Path('cats').joinpath('categories{}_{}.png'.format(agent, step))
    plt.savefig(str(new_path))
    plt.close()


def plot_category(category: NewCategory, stimuli: List, calculator: Calculator):
    xs = calculator.domain()
    plt.plot(xs, category.discretized_distribution(calculator), 'o', xs, category.discretized_distribution(calculator),
             '--')

    additional_ys = [-1 for _ in range(len(stimuli))]
    plt.plot(stimuli, additional_ys, marker='o', linestyle='None', markersize=10, color='red', label='Dodatkowe punkty')

    plt.legend(['data', 'cubic'], loc='best')
    plt.show()


def bokeh_plot_category(category: NewCategory, domain: List, quasi_pdf: List[float], active_stimuli=None):
    if active_stimuli is None:
        active_stimuli = []
    p = figure(title='Category Density', x_axis_label='support', y_axis_label='QuasiDensity')
    p.line(domain, quasi_pdf, legend_label=str(category), line_width=2)
    p.circle(active_stimuli, -.05, legend_label='Word X Category extension', color='red', size=1)
    return p


def plot_successes(steps: int, cs1: List[List[float]], cs2: List[List[float]], ds: List[List[float]],
                   cs12: List[List[float]]):
    fig, ax1 = plt.subplots()
    plt.ylim(bottom=0)
    plt.ylim(top=100)
    plt.xlabel("step")
    plt.ylabel("success")

    xs = range(steps)
    for values in [cs1, cs2, ds, cs12]:
        plt.plot(xs, values, 'r-', linewidth=0.6)
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('|active lexicon|')
    fig.tight_layout()
    plt.show()
