import os

from setuptools import setup, find_packages
from setuptools.command.install import install

from pyiwe import __version__ as pyiwe_ver
from pyiwe.tnt_install import TNTSetup


class _TNTSetup(install):
    """Install terminal TNT in setup.py"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tnt_setup = TNTSetup(installers_path=os.path.join('.', 'pyiwe', 'tnt_install'))

    def run(self):
        self.tnt_setup.setup()
        install.run(self)


with open("README.rst", "r", encoding='utf8') as f:
    long_description = f.read()

setup(
    name=f'pyiwe',
    version=pyiwe_ver,
    packages=find_packages(include=['pyiwe', 'pyiwe.utils', 'pyiwe.tnt_install']),
    url='',
    license='MIT',
    author='Alexander Popkov',
    author_email='alr.popkov@gmail.com',
    description="Python wrapper for TNT (Tree analysis using New Technology) implied weighting with clades support",
    long_description=long_description,
    install_requires=[
        'ete3>=3.1.2',
        'biopython>=1.78',
        'tqdm>=4.62.3',
        'numpy>=1.19',
        'pandas>=1.3.5',
        'pillow==8.4.0',
        'seaborn>=0.11.2',
        'matplotlib>=3.5.0',
        'jupyter',
    ],
    package_data={'pyiwe': ['tests/testdata/bryocorini/*',
                            'tests/testscripts/*',
                            'tnt_scripts/*',
                            'tnt_install/*'
                            ]},
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    python_requires='>=3.6',
    cmdclass={
        'install': _TNTSetup,
    },
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ]

)
