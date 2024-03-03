import importlib.metadata as importlib_metadata
import importlib.util


def is_torch_available():
    package_exists = importlib.util.find_spec("torch") is not None

    # Check we're not importing a "torch" directory somewhere but the actual library by
    # trying to grab the version
    if package_exists:
        try:
            _ = importlib_metadata.metadata("torch")
            return True
        except importlib_metadata.PackageNotFoundError:
            return False


if is_torch_available():
    from .trainer import BaseTrainer
else:
    # Inherits from a dummy `object` if torch is not available, so that python
    # succeeds to import this file.
    # BaseTrainer abstraction code will never inherit this dummy object as it checks if
    # torch is available.
    from builtins import object as BaseTrainer
