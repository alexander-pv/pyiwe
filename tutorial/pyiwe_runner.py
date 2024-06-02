"""
This is an example of a quick start with pyiwe package via terminal.
"""

import argparse
import os
import sys

import pandas as pd
from ete3 import Tree as ETree

import pyiwe
import pyiwe.utils.visualize as tree_vis


def parse_arguments() -> argparse.Namespace:
    """
    Parse arguments for TNT-based implied weighting branch support
    Args: None
    Example:
            $ python ./pyiwe_runner.py
    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Argument parser for pyiwe_runner.py')
    parser.add_argument('feature_matrix', metavar='feat_matrix', type=str,
                        help='str, path to the feature matrix for TNT')

    parser.add_argument('-k_start', metavar='k_start', type=float, default=1e-2,
                        help='float, minimum value in a linear scale or a degree in a logarithmic scale, default=1e-2')
    parser.add_argument('-k_stop', metavar='k_stop', type=float, default=1.5,
                        help='float, maximum value in a linear scale or a degree in a logarithmic scale, default=1.5')
    parser.add_argument('-k_num', metavar='k_num', type=int, default=100,
                        help='int, number of samples to generate, default=100')

    parser.add_argument('-k_scale', metavar='k_scale', type=str, default='log', choices=('log', 'lin'),
                        help='str, scale of concavity values, `log` or `linear`, default=`log`')
    parser.add_argument('-n_runs', metavar='n_runs', type=int, default=3,
                        help='int, the number of repeated IW runs, default=3')
    parser.add_argument('-cutoff', metavar='cutoff', type=float, default=0.5,
                        help='float, cutoff value between 0.0 and 1.0 for a final majority rule tree, default=0.5')
    parser.add_argument('-xmult_hits', metavar='xmult_hits', type=int, default=5,
                        help='int, produce N hits to the best length and stop, default=5')
    parser.add_argument('-xmult_level', metavar='xmult_level', type=int, default=3,
                        help='int, set level of search (0-10). Use 0-2 for easy data, default=3')
    parser.add_argument('-xmult_drift', metavar='xmult_drift', type=int, default=5,
                        help='int, cycles of drifting;, default=5')
    parser.add_argument('-hold', metavar='hold', type=int, default=500,
                        help='int, a tree buffer to keep up to specified number of trees, default=500')
    parser.add_argument('-output_folder', metavar='output_folder', type=str, default=os.path.join('.', 'output'),
                        help='str, path to store data, default=./output')

    parser.add_argument('-log_base', metavar='log_base', type=float, default=10.0,
                        help='float, base for calculating a log space for concavity constants, default=10.0')
    parser.add_argument('-float_prec', metavar='float_prec', type=int, default=5,
                        help='int, Floating point calculations precision, default=5')
    parser.add_argument('-tnt_seed', metavar='tnt_seed', type=str, default='1',
                        help='str, random seed properties for TNT, default=`1`')
    parser.add_argument('-seed', metavar='seed', type=int, default=42,
                        help='str, random seed for Python numpy, default=42')
    parser.add_argument('-tnt_echo', metavar='tnt_echo', type=str, choices=('-', '='), default='-',
                        help='str, `=`, echo each command, `-`, don`t echo, default=`-`')
    parser.add_argument('-memory', metavar='memory', type=float, default=1024 * 10,
                        help=f'float, Memory to be used by macro language, in KB, default={1024 * 10}')

    parser.add_argument('-c', action='store_true',
                        help='bool, clear temp *.tre files in output folder after processing')
    parser.add_argument('-v', action='store_true',
                        help='bool, add processing verbosity')

    return parser.parse_args()


def prepare_info_table(d: dict) -> pd.DataFrame:
    df_tree = pd.DataFrame(d).T
    df_tree['child_nodes'] = df_tree['clade'].apply(lambda x: len(x))
    df_tree['clade_id'] = df_tree.index
    return df_tree


def open_ete_browser(tree: ETree):
    tree.show()


def runner(args: argparse.Namespace):
    pyiwe_instance = pyiwe.PyIW(
        k_start=args.k_start,
        k_stop=args.k_stop,
        k_num=args.k_num,
        k_scale=args.k_scale,
        n_runs=args.n_runs,
        x_mult={
            'hits': args.xmult_hits,
            'level': args.xmult_level,
            'drift': args.xmult_drift
        },
        hold=args.hold,
        output_folder=args.output_folder,
        log_base=args.log_base,
        float_prec=args.float_prec,
        tnt_seed=args.tnt_seed,
        seed=args.seed,
        tnt_echo=args.tnt_echo,
        memory=args.memory,
        clear_output=args.c,
        verbose=args.v
    )

    majority_tree, k_alloc = pyiwe_instance.get_iw_branch_support(matrix_path=args.feature_matrix, cutoff=0.5)
    df = prepare_info_table(d=k_alloc)
    df.to_csv(os.path.join(args.output_folder, 'final_tree_clades_info.csv'), index=False)
    with open(os.path.join(args.output_folder, 'final_tree_newick.txt'), 'w') as f:
        f.write(majority_tree.fmt(format='newick'))
    open_ete_browser(tree=ETree(majority_tree.fmt(format='newick')))

    if int(pd.__version__.split('.')[1]) < 3:
        df_tree_flat = df.explode('k_vals')
    else:
        df_tree_flat = df.explode(column=['k_vals'])

    tree_vis.make_k_stripplot(df=df_tree_flat, x='clade_id', y='k_vals', save=False)


def main():
    runner(parse_arguments())


if __name__ == '__main__':
    sys.exit(main())
