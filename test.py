from __future__ import division  # force python 3 division in python 2
import matplotlib.pyplot as plt
from numpy import column_stack
from numpy import linspace
from numpy import arange
from numpy import array
from numpy import zeros
from numpy import amax
from numpy import random
import gibberish as gib


g = gib.Gibberish()

l100x100 = random.uniform(0,1,10000)
m100x100 = array(l100x100).reshape(100,100)
lex100 = [g.generate_word() for w in range(0,100)]

l50x50 = random.uniform(0,1,2500)
m50x50 = array(l50x50).reshape(50,50)
lex50 = [g.generate_word() for w in range(0,50)]


def plot(m,l):
    lexicon = l
    lxc = m
    n_rows = lxc.shape[0]
    n_categories = lxc.shape[1]
    fig, ax = plt.subplots()
    lxc_ex = column_stack((lxc, linspace(amax(lxc), 0, n_rows)))
    n_cols = lxc_ex.shape[1]
    #im = ax.imshow(lxc_ex, extent = (0.5, 10*n_cols, 10*n_rows, -0.5), aspect='auto')
    im = ax.imshow(lxc_ex, aspect="auto")
    left,right,bottom,top = im.get_extent()
    print("l=%f, r=%f, b=%f, t=%f" % (left,right,bottom,top))
    # We want to show all ticks...
    #ax.set_xticks(linspace(0, right, n_cols, endpoint=False))
    #ax.set_yticks(linspace(0,bottom, n_rows, endpoint=False))

    ax.set_xticks(arange(n_cols))
    ax.set_yticks(arange(n_rows))

    # ... and label them with the respective list entries
    x_tick_labels = [str(j + 1) for j in arange(n_categories)]
    x_tick_labels.append("s")
    ax.set_xticklabels(x_tick_labels, fontdict={'fontsize':6})
    for t in range(len(lexicon), n_rows):
        x_tick_labels.append('-')
    ax.set_yticklabels(lexicon, fontdict={'fontsize':6})
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
    plt.savefig("./simulation_results/matrices/matrix%d_%d" % (lxc.shape[0], lxc.shape[1]))
    plt.close()

plot(m100x100,lex100)