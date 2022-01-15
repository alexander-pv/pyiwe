import io
import re

from Bio import File
from Bio import Phylo as ph

supported_formats = {
    "newick": ph.NewickIO,
    "nexus": ph.NexusIO,
    "phyloxml": ph.PhyloXMLIO,
    "nexml": ph.NeXMLIO,
}


def check_tnt_tree_format(tnt_str: str):
    """
    Args:
        tnt_str: str, path to TNT-tree
    Returns:

    """
    data_fmt = tnt_str.split('.')[-1]
    if 'tre' not in data_fmt:
        raise AssertionError(f'Unknown data format: {data_fmt}')


def check_tree_formats_selection(tree_format: str):
    """
    Args:
        tree_format: str, tree format

    Returns:

    """
    _tree_formats = list(supported_formats.keys())
    if tree_format not in _tree_formats:
        raise AssertionError(f'Tree format is not recognized: {tree_format}\n Available: {_tree_formats}')


def _read_tnt_tree(path: str) -> list:
    """
    Parse TNT-generated tree
    Args:
        path: str, path to TNT-tree
    Returns: str

    """
    check_tnt_tree_format(path)
    with open(path, 'r') as f:
        tree_file = f.readlines()
    return tree_file


def _convert_tnt_treeline(line: str) -> str:
    """
    Prepare TNT-tree line
    Args:
        line: str, a row or line from TNT-tree file

    Returns: str
    """
    line = line.strip().strip(";")
    line = line.replace(".", "_")
    line = re.sub(r"([\w|\d])\s([\w|(])", r"\1, \2", line)
    line = re.sub(r"\)\(", r"), (", line)
    return line


def _convert_tnt_tree(tree: list) -> str:
    """
    Convert TNT-tree to Newick for Biopython.Phylo
    Args:
        tree: list, number of rows of TNT-tree file

    Returns: list

    """
    for i, line in enumerate(tree):
        newline = _convert_tnt_treeline(line)
        tree[i] = newline
    tree = ''.join(tree)
    tree = tree[tree.find('('):]
    tree = tree[tree.find('('):].replace('  ', ', ') + ';'
    return tree


def buffer_write(trees: list, tree_fmt: str, **kwargs) -> io.StringIO:
    """
    Write a sequence of trees to file in the given format to StringIO buffer.
    Args:
        trees:       list of trees
        string_io:   StringIO buffer
        tree_fmt:    str, tree format
        **kwargs:

    Returns:

    """

    if isinstance(trees, (ph.BaseTree.Tree, ph.BaseTree.Clade)):
        # Passed a single tree instead of an iterable -- that's OK
        trees = [trees]
    string_io = io.StringIO()
    with File.as_handle(string_io, "w+") as fp:
        getattr(supported_formats[tree_fmt], "write")(trees, fp, **kwargs)
    return string_io


def buffer_convert(trees, out_format: str, **kwargs) -> io.StringIO:
    """
    Convert between two tree file formats.
    Args:
        trees: ph.BaseTree.Tree, BaseTree.Clade
        out_format: str
        **kwargs:

    Returns: io.StringIO() buffer

    """
    return buffer_write(trees, out_format, **kwargs)


def get_tnt_trees(tree_path: str, output_format: str = 'newick',
                  verbose: bool = True) -> tuple[list[ph.BaseTree.Tree], str]:
    """
    Read TNT tree and return it in format `output_format`.
    Args:
        tree_path:     str, the path to a tree
        output_format: str, possible formats: 'newick', 'nexus', 'nexml', 'phyloxml', newick is default
        verbose:       bool, verbosity for trees reading

    Returns: tuple[list[ph.BaseTree.Tree], str]

    """
    check_tree_formats_selection(output_format)
    raw_tree = _read_tnt_tree(tree_path)[2:-1]
    raw_trees = _convert_tnt_tree(raw_tree)
    output_trees = [ph.read(io.StringIO(t), 'newick') for t in raw_trees.split('*')]

    if output_format != 'newick':
        new_output_trees_io = [buffer_convert(t, output_format) for t in output_trees]
        output_trees = [ph.read(io.StringIO(t.getvalue()), output_format) for t in new_output_trees_io]
    if verbose:
        n_nodes = [len(t.get_terminals()) for t in output_trees]
        print('Found %d trees with number of nodes: %s' % (len(output_trees), n_nodes))
    return output_trees, raw_trees
