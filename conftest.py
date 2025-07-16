import os
import nntool


def pytest_sessionstart(session):
    """Print the nntool package path being used for testing.

    This helps verify which installation of nntool is being used during test execution.
    """
    print(
        f"\nTesting nntool from: {os.path.dirname(nntool.__file__)} with version: {nntool.__version__}\n"
    )
