import importlib.metadata
import importlib.util
from typing import Tuple, Union


def _is_package_available(
    pkg_name: str, return_version: bool = False
) -> Union[Tuple[bool, str], bool]:
    """
    This function checks if a package is available and optionally returns its version and is borrowed from
    https://github.com/huggingface/transformers/blob/a847d4aa6bd2279f5be235dc0fd862f58f7403d1/src/transformers/utils/import_utils.py#L42
    """
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # TODO: Once python 3.9 support is dropped, `importlib.metadata.packages_distributions()`
            # should be used here to map from package name to distribution names
            # e.g. PIL -> Pillow, Pillow-SIMD; quark -> amd-quark; onnxruntime -> onnxruntime-gpu.
            # `importlib.metadata.packages_distributions()` is not available in Python 3.9.

            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            elif pkg_name == "quark":
                # TODO: remove once `importlib.metadata.packages_distributions()` is supported.
                try:
                    package_version = importlib.metadata.version("amd-quark")
                except Exception:
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


def cycle_dataloader(dl):
    while True:
        for data in dl:
            yield data


def divisible_by(numer, denom):
    return (numer % denom) == 0
