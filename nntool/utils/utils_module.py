import os
import datetime
import warnings

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
    else:
        warnings.warn(
            f"OUTPUT_PATH is not found in environment variables. Using path: {output_path}"
        )
        current_time = get_current_time()
        if append_date:
            output_path = os.path.join(output_path, current_time)

    return output_path, current_time
