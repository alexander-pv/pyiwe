from __future__ import annotations
import os
import sys


class PyIWConfig:

    def __init__(self):
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.available_platforms = ('win32', 'linux')
        self.platform = sys.platform
        self.tnt_path = self.get_tnt_path()
        self.correct_platfrom = self.platform in self.available_platforms

    def get_tnt_path(self) -> str:
        split_symb = {'linux': ":", 'win32': ";"}
        found_path = None
        for p in os.environ["PATH"].split(split_symb.get(self.platform)):
            if "TNT" in p:
                found_path = p
                break
        return found_path


pyiw_config = PyIWConfig()
