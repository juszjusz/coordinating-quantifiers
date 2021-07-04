import argparse

import data_postprocess
import simulation

parser = argparse.ArgumentParser(prog='quantifiers simulation')

parser.add_argument('--simulation_name', '-sn', help='simulation name', type=str, default='test-x')
parser.add_argument('--population_size', '-p', help='population size', type=int, default=4)
parser.add_argument('--stimulus', '-stm', help='quotient or numeric', type=str, default='quotient')
parser.add_argument('--max_num', '-mn', help='max number for numerics or max denominator for quotients', type=int,
                    default=100)
parser.add_argument('--discriminative_threshold', '-dt', help='discriminative threshold', type=float, default=.95)
parser.add_argument('--delta_inc', '-dinc', help='delta increment', type=float, default=.2)
parser.add_argument('--delta_dec', '-ddec', help='delta decrement', type=float, default=.2)
parser.add_argument('--delta_inh', '-dinh', help='delta inhibition', type=float, default=.2)
parser.add_argument('--alpha', '-a', help='forgetting rate', type=float, default=.01)
parser.add_argument('--super_alpha', '-sa', help='complete forgetting of categories that have smaller weights',
                    type=float, default=.001)
parser.add_argument('--beta', '-b', help='learning rate', type=float, default=0.2)
parser.add_argument('--steps', '-s', help='number of steps', type=int, default=50)
parser.add_argument('--runs', '-r', help='number of runs', type=int, default=1)
parser.add_argument('--guessing_game_2', '-gg2', help='is the second stage of the guessing game on', type=bool,
                    default=False)
parser.add_argument('--load_simulation', '-l', help='load and rerun simulation from pickled simulation step',
                    type=str)
# parser.add_argument('--parallel', '-pl', help='run parallel runs', type=bool, default=True)
parser.add_argument('--in_mem_calculus_path', '-path', help='path to precomputed integrals', type=str,
                    default='inmemory_calculus')

parser.add_argument('--data_root', '-d', help='root path to {data, cats, langs, matrices, ...}', type=str,
                    default="test-x")
parser.add_argument('--plot_cats', '-pc', help='plot categories', type=bool, default=True)
parser.add_argument('--plot_langs', '-pl', help='plot languages', type=bool, default=True)
parser.add_argument('--plot_langs2', '-pl2', help='plot languages 2', type=bool, default=True)
parser.add_argument('--plot_matrices', '-pm', help='plot matrices', type=bool, default=False)
parser.add_argument('--plot_success', '-ps', help='plot success', type=bool, default=False)
parser.add_argument('--plot_mon', '-mon', help='plot monotonicity', type=bool, default=False)
parser.add_argument('--plot_mons', '-mons', help='plot monotonicity', type=str, nargs='+', default='')
parser.add_argument('--plot_conv', '-conv', help='plot convexity', type=str, nargs='+', default='')
parser.add_argument('--plot_num_DS', '-nds', help='plot success', type=bool, default=False)
parser.add_argument('--hdf_franek', '-hdf', help='hdf franek', type=bool, default=False)
parser.add_argument('--parallelism', '-ppp', help='number of processes (unbounded if 0)', type=int, default=8)

# simulation.start(parsed_params=vars(parser.parse_args()))
data_postprocess.start(parsed_params=vars(parser.parse_args()))
