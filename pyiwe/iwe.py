import glob
import itertools
import os
import subprocess
from typing import TextIO

import numpy as np
import tqdm
from Bio import Phylo as ph
from Bio.Phylo import Consensus

from . import config
from .utils import common as common
from .utils import processing as tree_proc


class PyIW:
    """
    A wrapper for implied weighting with TNT
    """

    def __init__(self, k_start: float,
                 k_stop: float,
                 k_num: int,
                 x_mult: dict,
                 k_scale: str = 'log',
                 output_folder: str = '../output',
                 clear_output: int = False,
                 n_runs: int = 1,
                 hold: int = 500,
                 float_prec: int = 5,
                 log_base: float = 10.0,
                 noised_k=True,
                 iw_iter_consensus: str = 'strict',
                 iw_iter_consensus_kwargs: dict or None = None,
                 tnt_seed: str = '1',
                 tnt_echo: str = '-',
                 seed: int = 42,
                 memory: float = 1024 * 10,
                 verbose: bool = True):
        """
        TNT implied weighting parameters.
        Note, TNT scripting language offers more parameters. See TNT help.
        Args:
            k_start:        float, minimum value in a linear scale or a degree in a logarithmic scale;
            k_stop:         float, maximum value in a linear scale or a degree in a logarithmic scale;
            k_num:          int, number of concavity constants to generate;

            x_mult: dict, run multiple replications, using sectorial searches, drifting, ratchet and fusing combined;
                    Keys:
                    hits,      int, produce N hits to the best length and stop;
                    level,     int, set level of search (0-10). Use 0-2 for easy data;
                    drift,     int, cycles of drifting;

            k_scale:        str, scale of concavity values, 'log' or 'linear';
            output_folder:  str, path to store data;
            clear_output:   bool, clear temp files in output folder after processing;
            n_runs:         int, the number of repeated IW runs;
            hold:           int, a tree buffer to keep up to specified number of trees;
            float_prec: int, Floating point calculations precision:
                                 If floating point enabled, uses N decimal digits when printing
                                 floats (0 uses no digits; equivalent to integer only, but uses more memory);

            log_base: float, base for calculating a log space;
            noised_k: bool,  adjust k value at each iteration on the next runs;

            iw_iter_consensus: str, consensus type to merge trees output after each IW iteration.
                               Options: 'strict'(default), 'majority', 'adam';

            iw_iter_consensus_kwargs: dict, optional kwargs for iw_iter_consensus 'majority' or 'adam';

            tnt_seed: str, random seed properties for TNT.
                    Options:
                    N    set random seed as N ( 0 = time ; default = 1 )
                    +N   increase random seed by N
                    *    set a new random seed, at random
                    [    in wagner trees, randomize insertion sequence
                    ]    in wagner trees, try insertions for new taxa from
                       top to bottom or from bottom up (=default)
                    >    in wagner trees, also randomize outgroup.  This
                       cannot be done when there are constraints or
                       asymmetric Sankoff characters (randomization is
                       skipped). Note that some "xmult" options use
                       internal constraints (and then skip randomization)
                    <    in wagner trees, outgroup is always the first taxon
                       placed in the tree (=default)
                    :N  in multiple randomizations, instead of making sure
                       that each new seed is different from the ones used
                       before, increase the seed by N.  This may save time
                       in very extensive randomizations (where checking
                       previous seeds takes time).  When N=0, checks previous
                       seeds (this the default).
                    !    use quick approximation for randomization (faster)
                    -    use careful randomizations (slower, more random; default)

            tnt_echo: str, `=`, echo each command, `-`, don't echo;

            memory:  float, Memory to be used by macro language, in KB;
            verbose: bool, add processing verbosity;
        """
        if iw_iter_consensus_kwargs is None:
            iw_iter_consensus_kwargs = {}
        self.k_start = k_start
        self.k_stop = k_stop
        self.k_num = k_num
        self.x_mult = x_mult
        self.output_folder = output_folder
        self.clear_output = clear_output
        self.k_scale = k_scale
        self.n_runs = n_runs
        self.hold = hold
        self.float_prec = float_prec
        self.noised_k = noised_k
        self.noised_k_scale = 1e-3
        self.tnt_seed = tnt_seed
        self.tnt_echo = tnt_echo
        self.seed = seed
        self.memory = memory
        self.verbose = verbose

        self.iw_iter_consensus = iw_iter_consensus
        self.iw_iter_consensus_kwargs = iw_iter_consensus_kwargs

        if self.k_scale == 'log':
            self.k_range = np.round(
                np.logspace(start=self.k_start, stop=self.k_stop, num=self.k_num, base=log_base), self.float_prec)
        elif self.k_scale == 'lin':
            self.k_range = np.round(
                np.linspace(start=self.k_start, stop=self.k_stop, num=self.k_num), self.float_prec)
        else:
            raise ValueError(f'Possible values for scale parameter: {self.k_scale}')

        self._check_params()
        self._check_output_folder()
        self.noised_k_vectorized = np.vectorize(self._make_k_noised)
        self.logger = Logger(log_path=self.output_folder)

        self.tnt_script_path = os.path.join(config.pyiw_config.root_dir, 'tnt_scripts', 'iwe_template.tnt')

    def _clear_temp_output(self):
        trees_paths = glob.glob(os.path.join(self.output_folder, 'temp_*_k_*.tre'))
        for tree_path in trees_paths:
            try:
                os.remove(tree_path)
            except OSError:
                print(f"[{self.__class__.__name__}]Error deleting file: {tree_path}")

    def _check_output_folder(self) -> None:
        os.makedirs(self.output_folder, exist_ok=True)
        for filename in os.listdir(self.output_folder):
            if filename.split('.')[-1] in ('tre', 'txt'):
                os.remove(os.path.join(self.output_folder, filename))

    def _check_params(self) -> None:
        assert self.k_start > 0
        assert self.k_stop > self.k_start
        assert self.k_range[-1] <= 1e3

    def _check_nodes(self, trees: list[ph.BaseTree.Tree]) -> None:
        nodes_cnt = (t.get_terminals() for t in trees)
        if np.unique(nodes_cnt).size != 1:
            raise AssertionError(f'The number of nodes for different trees files are different: {nodes_cnt}')
        if self.verbose:
            print(f'[{self.__class__.__name__}]The number of nodes in the merged trees:{nodes_cnt[0]}')

    def _make_k_noised(self, k: float):
        eps = np.random.RandomState(self.seed).normal(size=1, loc=0, scale=self.noised_k_scale)
        if k + eps > 1e3 or k + eps < 0:
            k -= eps
        else:
            k += eps
        return k

    def _prepare_command(self, matrix_path: str, k: float, run: int) -> str:
        """
        Prepare IW command for TNT.
        iwe_template.tnt positional arguments:
            %1:  Concavity constant k;
            %2:  A tree buffer to keep up to specified number of trees;
            %3:  Produce N hits to the best length and stop;
            %4:  Level of search (0-10). Use 0-2 for easy data;
            %5:  Cycles of drifting;
            %6:  Floating point calculations precision;
            %7:  Random seed value;
            %8:  Memory to be used by macro language, in KB;
            %9:  The current run id with the same concavity constant k;
            %10: Echo off/on;
            %11: Path to the temp files and log output;
        Args:
            matrix_path: str, path to the feature matrix;
            k:           float, concavity parameter for IW;
            run:         int, the number of IW run with the same concavity constant;
        Returns: str
        """
        command = ("tnt", "proc", matrix_path,
                   "run", f"\"{self.tnt_script_path}",
                   k,
                   self.hold,
                   self.x_mult['hits'],
                   self.x_mult['level'],
                   self.x_mult['drift'],
                   self.float_prec,
                   self.tnt_seed,
                   self.memory,
                   run,
                   self.tnt_echo,
                   self.output_folder + ';"'
                   )
        return " ".join([str(x) for x in command])

    @common.timing
    def run_iw(self, matrix_path: str) -> None:
        """
        Run IW with specified parameters.
        Args:
            matrix_path: str, path to the feature matrix;
        Returns: None
        """
        self.check_feature_matrix(matrix_path)
        for run in range(self.n_runs):
            self.k_range = self.noised_k_vectorized(self.k_range) if self.noised_k and run else self.k_range
            pbar = tqdm.tqdm(enumerate(self.k_range), total=self.k_range.size)
            for i, k in pbar:
                pbar.set_description(desc='Run: %d K value: %f' % (run, k))
                cmnd = self._prepare_command(matrix_path=matrix_path, k=k, run=run)
                if self.verbose:
                    print(f'[{self.__class__.__name__}]TNT command:', cmnd)
                self._execute_terminal(cmnd)
                self.logger.update(desc=f'Run: {run} Iteration: {i}')
        self.logger.close_log()

    def load_iw_trees(self) -> list[ph.BaseTree.Tree]:
        """
        Load trees for each IW iteration and perform consensus for each batch of trees.
        Returns: dict, {index_0: {'k': concavity value,'tree': merged trees},
                        index_1: {...},
                        index_2: {...},
                        ...}
        """
        trees_paths = glob.glob(os.path.join(self.output_folder, 'temp_*_k_*.tre'))
        trees = []
        for i, tp in enumerate(trees_paths):
            iw_trees = tree_proc.get_tnt_trees(tree_path=tp, output_format='newick', verbose=self.verbose)[0]
            merged_tree = self.merge_iw_trees(iw_trees)
            merged_tree = self.set_k_to_tree(merged_tree, float(tp.split('k_')[1].replace('.tre', '')))
            trees.append(merged_tree)
        return trees

    def merge_iw_trees(self, trees_list: list[ph.BaseTree.Tree]) -> ph.BaseTree.Tree:
        """
        Merge trees in trees_list
        Args:
            trees_list: list, list of trees ph.BaseTree.Tree of a specific concavity value k
        Returns: merged ph.BaseTree.Tree
        """
        consensus = {
            'strict': Consensus.strict_consensus,
            'majority': Consensus.majority_consensus,
            'adam': Consensus.adam_consensus
        }
        if self.iw_iter_consensus not in list(consensus.keys()):
            raise AssertionError(f'Available consensuses for each IW iteration: {list(consensus.keys())}')
        return consensus[self.iw_iter_consensus](trees_list, **self.iw_iter_consensus_kwargs)

    def get_iw_branch_support(self, matrix_path: str, cutoff: float,
                              support_precision: int = 1) -> tuple[ph.BaseTree.Tree, dict[int: dict]]:
        """
        Get a target tree with IW branch support and a dictionary of k distributions across nodes in that target tree.
        Args:
            matrix_path:       str,
            cutoff:            float,
            support_precision: int,
        Returns: tuple[ph.BaseTree.Tree, dict[str: list[float]]]

        """
        self.run_iw(matrix_path=matrix_path)
        merged_trees = self.load_iw_trees()
        majority_tree = _iw_maj_consensus(merged_trees, cutoff=cutoff)
        for c in majority_tree.get_nonterminals():
            if c.confidence:
                c.confidence = round(c.confidence, support_precision)
        clades_k_alloc = self.get_clade_k_allocations(majority_tree)
        if self.clear_output:
            self._clear_temp_output()
        return majority_tree, clades_k_alloc

    @staticmethod
    def _execute_terminal(command: str) -> None:
        subprocess.run(command, shell=True)

    def check_feature_matrix(self, path: str) -> None:
        if self.verbose:
            print(f'[{self.__class__.__name__}]Checking path: {path}')
        prohibited_symbols = ('(', ')')
        found = 0
        with open(path, 'r') as f:
            feature_matrix_txt = ' '.join(f.readlines())
        for s in prohibited_symbols:
            if s in feature_matrix_txt:
                feature_matrix_txt = feature_matrix_txt.replace(s, '_')
                found = 1
        if found:
            with open(path, 'w') as f:
                f.write(feature_matrix_txt)

    @staticmethod
    def set_k_to_tree(tree: ph.BaseTree.Tree, k: float) -> ph.BaseTree.Tree:
        tree.k = k
        for clade in tree.find_clades(terminal=False):
            clade.k = k
        return tree

    @staticmethod
    def get_clade_k_allocations(tree: ph.BaseTree.Tree) -> dict[str: int]:
        k_alloc = {
            i:
                {
                    'clade': tuple(t.name for t in c.get_terminals()),
                    'k_vals': tuple(map(float, c.k_vals_str.split(';'))),
                    'confidence': c.confidence
                }
            for i, c in enumerate(tuple(tree.find_clades(terminal=False)))
        }
        return k_alloc


def _iw_count_clades(trees: list[ph.BaseTree.Tree]) -> tuple[dict, int]:
    """Count distinct clades (different sets of terminal names) in the trees (PRIVATE).

    Return a tuple first a dict of bitstring (representing clade) and a tuple of its count of
    occurrences and sum of branch length for that clade, second the number of trees processed.

    Modified for additional information to store concavity constants k of implied weighting method.
    Source: https://github.com/biopython/biopython

    :Parameters:
        trees : iterable
            An iterable that returns the trees to count

    """
    bitstrs = {}
    tree_count = 0
    for tree in trees:
        assert hasattr(tree, 'k')
        tree_count += 1
        clade_bitstrs = Consensus._tree_to_bitstrs(tree)
        for clade in tree.find_clades(terminal=False):
            bitstr = clade_bitstrs[clade]
            if bitstr in bitstrs:
                count, sum_bl, k_vals = bitstrs[bitstr]
                count += 1
                sum_bl += clade.branch_length or 0
                k_vals.append(clade.k)
                bitstrs[bitstr] = (count, sum_bl, k_vals)
            else:
                bitstrs[bitstr] = (1, clade.branch_length or 0, [clade.k])
    return bitstrs, tree_count


def _iw_maj_consensus(trees: list[ph.BaseTree.Tree], cutoff: float = 0) -> ph.BaseTree.Tree:
    """Search majority rule consensus tree from multiple trees.

    This is a extend majority rule method, which means the you can set any
    cutoff between 0 ~ 1 instead of 0.5. The default value of cutoff is 0 to
    create a relaxed binary consensus tree in any condition (as long as one of
    the provided trees is a binary tree). The branch length of each consensus
    clade in the result consensus tree is the average length of all counts for that clade.

    Modified for an additional infoprmation to store concavity constants k of implied weighting method.
    Source: https://github.com/biopython/biopython

    :Parameters:
        trees : iterable
            iterable of trees to produce consensus tree.

    """
    tree_iter = iter(trees)
    first_tree = next(tree_iter)

    terms = first_tree.get_terminals()
    bitstr_counts, tree_count = _iw_count_clades(itertools.chain([first_tree], tree_iter))

    # Sort bitstrs by descending #occurrences, then #tips, then tip order
    bitstrs = sorted(
        bitstr_counts.keys(),
        key=lambda bitstr: (bitstr_counts[bitstr][0], bitstr.count("1"), str(bitstr)),
        reverse=True,
    )
    root = ph.BaseTree.Clade()
    root.k_vals_str = ';'.join((str(t.k) for t in trees))

    if bitstrs[0].count("1") == len(terms):
        root.clades.extend(terms)
    else:
        raise ValueError("Taxons in provided trees should be consistent")
    # Make a bitstr-to-clades dict and store root clade
    bitstr_clades = {bitstrs[0]: root}
    # create inner clades
    for bitstr in bitstrs[1:]:
        # apply majority rule
        count_in_trees, branch_length_sum, k_vals = bitstr_counts[bitstr]
        confidence = 100.0 * count_in_trees / tree_count
        if confidence < cutoff * 100.0:
            break
        clade_terms = [terms[i] for i in bitstr.index_one()]
        clade = ph.BaseTree.Clade()
        clade.clades.extend(clade_terms)
        clade.confidence = confidence
        clade.k_vals_str = ';'.join(map(str, k_vals))
        clade.branch_length = branch_length_sum / count_in_trees
        bsckeys = sorted(bitstr_clades, key=lambda bs: bs.count("1"), reverse=True)
        # check if current clade is compatible with previous clades and
        # record it's possible parent and child clades.
        compatible = True
        parent_bitstr = None
        child_bitstrs = []  # multiple independent childs
        for bs in bsckeys:
            if not bs.iscompatible(bitstr):
                compatible = False
                break
            # assign the closest ancestor as its parent
            # as bsckeys is sorted, it should be the last one
            if bs.contains(bitstr):
                parent_bitstr = bs
            # assign the closest descendant as its child
            # the largest and independent clades
            if (
                    bitstr.contains(bs)
                    and bs != bitstr
                    and all(c.independent(bs) for c in child_bitstrs)
            ):
                child_bitstrs.append(bs)
        if not compatible:
            continue

        if parent_bitstr:
            # insert current clade; remove old bitstring
            parent_clade = bitstr_clades.pop(parent_bitstr)
            # update parent clade childs
            parent_clade.clades = [
                c for c in parent_clade.clades if c not in clade_terms
            ]
            # set current clade as child of parent_clade
            parent_clade.clades.append(clade)
            # update bitstring
            # parent = parent ^ bitstr
            # update clade
            bitstr_clades[parent_bitstr] = parent_clade

        if child_bitstrs:
            remove_list = []
            for c in child_bitstrs:
                remove_list.extend(c.index_one())
                child_clade = bitstr_clades[c]
                child_clade.k_vals = k_vals
                parent_clade.clades.remove(child_clade)
                clade.clades.append(child_clade)
            remove_terms = [terms[i] for i in remove_list]
            clade.clades = [c for c in clade.clades if c not in remove_terms]
        # put new clade
        bitstr_clades[bitstr] = clade
        if (len(bitstr_clades) == len(terms) - 1) or (
                len(bitstr_clades) == len(terms) - 2 and len(root.clades) == 3
        ):
            break
    return ph.BaseTree.Tree(root=root)


class Logger:

    def __init__(self, log_path: str, tnt_experiment_log: str = 'logfile.txt', summary_log: str = 'tnt_logfile.txt'):
        self.tnt_experiment_log = tnt_experiment_log
        self.summary_log = summary_log
        self.log_path = log_path
        self.file = self.open_log()

    def open_log(self) -> TextIO:
        return open(os.path.join(self.log_path, self.summary_log), 'w')

    def close_log(self):
        self.file.close()

    def update(self, desc: str = ''):
        with open(os.path.join(self.log_path, self.tnt_experiment_log), 'r') as f:
            self.file.write("\n%s\n%s" % (desc, " ".join(f.readlines())))
        os.remove(os.path.join(self.log_path, self.tnt_experiment_log))
