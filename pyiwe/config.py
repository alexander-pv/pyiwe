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
        tnt_path = {
            'linux': os.environ.get('TNT_PATH'),
        }
        return tnt_path.get(self.platform)


pyiw_config = PyIWConfig()
