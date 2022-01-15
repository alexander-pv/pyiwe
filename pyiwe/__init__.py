"""
TNT (Tree analysis using New Technology) implied weighting with branch supports Python wrapper.
TNT source: http://www.lillo.org.ar/phylogeny/tnt/ (Goloboff, Farris, & Nixon, 2003)
==================================
See README.md for complete documentation.
"""

__version__ = '0.0.1'

from .iwe import PyIW
from .utils import processing
from .utils import visualize
from .config import pyiw_config
from .tnt_install import TNTSetup
TNTSetup().setup()

__all__ = ['iwe', 'utils', 'config']
