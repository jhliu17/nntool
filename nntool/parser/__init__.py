import cythonpackage

cythonpackage.init(__name__)
from .parser_module import parse_from_cli
