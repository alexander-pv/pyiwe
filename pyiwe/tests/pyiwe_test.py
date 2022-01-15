import os
import subprocess
import sys

from .. import config


def test_platform() -> None:
    available_platforms = ('win32', 'linux')
    if sys.platform not in available_platforms:
        raise AssertionError(f'Not implemented for the platform: {sys.platform}.\nAvailable: {available_platforms}')
    print(f'\nSuccessfully detected {sys.platform} platform!')


def test_tnt_installation() -> None:
    """
    Test TNT existence in the current platform.
    In Windows and Linux TNT path should be in 'PATH' environment variable.
    In both OS hello_world.tnt example is executed to check that TNT is installed correctly.
    Returns: None
    """
    _test_filepath = os.path.join(config.pyiw_config.root_dir,
                                  'tests', 'testscripts', 'hello_world.tnt')
    if 'TNT' in os.environ["PATH"]:
        try:
            subprocess.check_call(["tnt", "run", _test_filepath + ";"])
        except FileNotFoundError:
            print('\nFileNotFoundError:\n' +
                  'Detected `TNT` keyword in PATH but can not actually run TNT via your PATH environment variable.')
            check_command = {
                "linux": "echo $PATH",
                "win32": r"(Get-ItemProperty -Path 'Registry::" +
                         r"""HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment'""" +
                         r"-Name PATH).path"
            }
            print(f"\nPlease, check PATH in your system terminal:\n{check_command}\n")
    else:
        raise AssertionError('No TNT in PATH environment variable')
    print('TNT was successfully launched!')


def test_pyiw_run() -> None:
    import pyiwe
    _matrix_path = os.path.join(config.pyiw_config.root_dir,
                                'tests', 'testdata', 'bryocorini', 'SI_4_Bryocorinae_matrix.tnt')
    pyiwe = pyiwe.PyIW(
        k_start=1e-2,
        k_stop=1,
        k_num=5,
        x_mult={'hits': 5, 'level': 3, 'drift': 5},
        n_runs=2,
    )
    pyiwe.run_iw(matrix_path=_matrix_path)
