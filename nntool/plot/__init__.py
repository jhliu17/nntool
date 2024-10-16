import cythonpackage

cythonpackage.init(__name__)

from .csrc.plot_module import *
from .context import enable_latexify
