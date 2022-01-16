"""
TNT (Tree analysis using New Technology) implied weighting with branch supports Python wrapper.
TNT source: http://www.lillo.org.ar/phylogeny/tnt/ (Goloboff, Farris, & Nixon, 2003)
==================================
See README.md for complete documentation.
"""

__version__ = '0.0.3'

import sys
import traceback

from .tnt_install import TNTSetup

try:
    from .iwe import PyIW
    from .utils import processing
    from .utils import visualize
    from .config import pyiw_config
except ModuleNotFoundError:
    print('ModuleNotFoundError happened. Check your dependencies.')
    traceback.print_exc(file=sys.stdout)

TNTSetup().setup()

__all__ = ['iwe', 'utils', 'config']
