import cythonpackage

cythonpackage.init(__name__)


def test_import():
    print(f"nntool located at {__file__} is imported!")
