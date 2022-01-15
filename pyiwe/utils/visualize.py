import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Bio import Phylo as ph
from PIL import Image


def render_tree(tree: ph.BaseTree.Tree, filename: str = 'tree_plot.png', title: str = 'Tree',
                figsize: tuple[int, int] = (10, 8), save_path: str = '.', kwargs: dict or None = None):
    """
    Args:
        tree:      ph.BaseTree.Tree, tree;
        filename:  str, name for saving an image;
        title:     str, image title;
        figsize:   tuple, figure size;
        save_path: str, path to save an image;
        kwargs:    dict,
    Returns: None

    """
    kwargs = kwargs if kwargs else dict()
    if 'units' not in kwargs.keys():
        kwargs.update({'units': 'px'})
    tree.render(file_name=os.path.join(save_path, filename), **kwargs)
    image = Image.open(os.path.join(save_path, filename))
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.axis('off')
    plt.imshow(image)


def make_k_stripplot(df: pd.DataFrame, x: str, y: str, fontsize: int = 12,
                     title: str = "Concavity constants distribution for each tree clade",
                     y_label: str = "IW concavity constants", xlabel: str = "Clade ID",
                     save: bool = False, save_name: str = 'concavity_vals_stripplot', save_path: str = '.',
                     **kwargs) -> None:
    """
    Args:
        df:          pd.DataFrame,
        x:           str, clade_id column
        y:           str, IW concavity values column
        fontsize:    int,
        title:       str,
        y_label:     str,
        xlabel:      str,
        save:        bool,
        save_name:   str,
        save_path:   str,
        **kwargs:
    Returns: None

    """
    sns.set()
    ax = sns.stripplot(x=x, y=y, data=df, **kwargs)
    ax.set_ylabel(y_label, fontdict={"fontsize": fontsize})
    ax.set_xlabel(xlabel, fontdict={"fontsize": fontsize})
    plt.title(title, fontsize=fontsize)
    if save:
        plt.savefig(os.path.join(save_path, f"{save_name}.png"))
    plt.show()
