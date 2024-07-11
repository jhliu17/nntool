import os
import datetime
import warnings
import tomli

from functools import cache


def get_current_time():
    """get current time in this format: MMDDYYYY_HHMMSS

    :return: time in the format MMDDYYYY_HHMMSS
    """
    # Get the current time
    current_time = datetime.datetime.now()

    # Format the time (MDY_HMS)
    formatted_time = current_time.strftime("%m%d%Y_%H%M%S")

    return formatted_time


def read_toml_file(file_path: str) -> dict:
    """Read a toml file and return the content as a dictionary

    :param file_path: path to the toml file
    :return: content of the toml file as a dictionary
    """
    with open(file_path, "rb") as f:
        content = tomli.load(f)

    return content


@cache
def get_output_path(
    output_path: str = "./", append_date: bool = True
) -> tuple[str, str]:
    """Get output path based on environment variable OUTPUT_PATH.
    The output path is appended with the current time if append_date is True (e.g. /outputs/xxx/MMDDYYYY_HHMMSS).

    The result for the same input is cached.

    :param append_date: append a children folder with the date time, defaults to True
    :return: (output path, current time)
    """
    if "OUTPUT_PATH" in os.environ:
        output_path = os.environ["OUTPUT_PATH"]
        current_time = "" if not append_date else os.path.split(output_path)[-1].strip()
    elif "NNTOOL_OUTPUT_PATH" in os.environ:
        # reuse the NNTOOL_OUTPUT_PATH if it is set
        output_path = os.environ["NNTOOL_OUTPUT_PATH"]
        current_time = "" if not append_date else os.path.split(output_path)[-1].strip()
    else:
        current_time = get_current_time()
        if append_date:
            output_path = os.path.join(output_path, current_time)

        os.environ["NNTOOL_OUTPUT_PATH"] = output_path
        warnings.warn(
            f"OUTPUT_PATH is not found in environment variables. NNTOOL_OUTPUT_PATH is set using path: {output_path}"
        )

    return output_path, current_time
