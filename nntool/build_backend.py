import os

from setuptools import build_meta as _orig
from setuptools.build_meta import *


def add_cythonpackage(requires: list):
    new_list = list(requires)
    new_list.append("buildkit/cythonpackage")
    return new_list


def get_requires_for_build_wheel(config_settings=None):
    if not os.getenv("NNTOOL_PYTHON_BUILD"):
        # pop out "buildkit/cythonpackage"
        return add_cythonpackage(_orig.get_requires_for_build_wheel(config_settings))
    else:
        return _orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    if not os.getenv("NNTOOL_PYTHON_BUILD"):
        # pop out "buildkit/cythonpackage"
        return add_cythonpackage(_orig.get_requires_for_build_sdist(config_settings))
    else:
        return _orig.get_requires_for_build_sdist(config_settings)
