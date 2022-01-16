import os
import subprocess
import sys


class TNTSetup:
    """Install terminal TNT"""

    def __init__(self, installers_path: str or None = None):
        self.tnt_keyword = 'TNT'
        self.env_variable = "PATH"
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.installers_path = installers_path if installers_path else self.root_dir
        self.available_platforms = ('linux', 'win32')
        self.platform = sys.platform
        self.tnt_install_scripts = {
            'linux': ["bash", os.path.join(self.installers_path, 'install_tnt_nix.sh')],
            'win32': ["powershell.exe", os.path.join(self.installers_path, 'install_tnt_win.ps1')]
        }

    def setup(self):
        if self.platform not in self.available_platforms:
            raise AssertionError(f"Incorrect platform:{self.platform}." +
                                 f"Installation implemented for: {self.available_platforms}")
        if self.tnt_keyword not in os.environ[self.env_variable]:
            subprocess.call(self.tnt_install_scripts[self.platform], stdout=sys.stdout)
