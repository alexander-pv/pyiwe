Python wrapper for TNT (Tree analysis using New Technology) implied weighting with clades support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation
^^^^^^^^^^^^

1. Pip:

.. code:: bash

   $ pip install pyiwe

2. conda:

.. code:: bash

   $ conda install -c alexander-pv pyiwe

The package file from PyPi or conda does not include terminal TNT. To
install it, open python in terminal mode and import ``pyiwe`` package.

.. code:: bash

   $ python
   $ >>> import pyiwe

3. From source:

.. code:: bash

   $ git clone git@github.com:alexander-pv/pyiwe.git && cd pyiwe
   $ pip install .

Terminal TNT will be installed automatically.

Tutorial
^^^^^^^^

-  ``implied_weighting_theory.ipynb``, theory behind implied weighting
   with fitting functions plots to play;
-  ``pyiwe_example.ipynb``, examples of reading TNT trees, plotting
   trees, getting branch supports and concavity values distributions for
   each clade in a tree based on TNT feature matrices;
-  ``pyiwe_runner.py``, terminal-based example for a quick start;

Run ``pyiwe_runner.py`` to see arguments help:

.. code:: bash

   $ cd ./pyiwe/tutorials && python pyiwe_runner.py -h

::

   Argument parser for pyiwe_runner.py

   positional arguments:
     feat_matrix           str, path to the feature matrix for TNT

   optional arguments:
     -h, --help            show this help message and exit
     -k_start k_start      float, minimum value in a linear scale or a degree in a logarithmic scale, default=1e-2
     -k_stop k_stop        float, maximum value in a linear scale or a degree in a logarithmic scale, default=1.5
     -k_num k_num          int, number of samples to generate, default=100
     -k_scale k_scale      str, scale of concavity values, `log` or `linear`, default=`log`
     -n_runs n_runs        int, the number of repeated IW runs, default=3
     -cutoff cutoff        float, cutoff value between 0.0 and 1.0 for a final majority rule tree, default=0.5
     -xmult_hits xmult_hits
                           int, produce N hits to the best length and stop, default=5
     -xmult_level xmult_level
                           int, set level of search (0-10). Use 0-2 for easy data, default=3
     -xmult_drift xmult_drift
                           int, cycles of drifting;, default=5
     -hold hold            int, a tree buffer to keep up to specified number of trees, default=500
     -output_folder output_folder
                           str, path to store data, default=./output
     -log_base log_base    float, base for calculating a log space for concavity constants, default=10.0
     -float_prec float_prec
                           int, Floating point calculations precision, default=5
     -tnt_seed tnt_seed    str, random seed properties for TNT, default=`1`
     -seed seed            str, random seed for Python numpy, default=42
     -tnt_echo tnt_echo    str, `=`, echo each command, `-`, don`t echo, default=`-`
     -memory memory        float, Memory to be used by macro language, in KB, default=10240
     -c                    bool, clear temp *.tre files in output folder after processing
     -v                    bool, add processing verbosity

Basic example:

.. code:: bash

   $ cd ./pyiwe/tutorials
   $ python pyiwe_runner.py ../pyiwe/tests/testdata/bryocorini/SI_4_Bryocorinae_matrix.tnt -c

References
^^^^^^^^^^

-  TNT source: http://www.lillo.org.ar/phylogeny/tnt (Goloboff, Farris,
   & Nixon, 2003)
-  Biopython: https://biopython.org
-  ETE, Python Environment for Tree Exploration: http://etetoolkit.org
