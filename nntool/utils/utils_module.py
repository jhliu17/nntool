import os
import datetime
import warnings

from functools import cache


def get_current_time():
    # Get the current time
    current_time = datetime.datetime.now()

    # Format the time (MDY_HMS)
    formatted_time = current_time.strftime("%m%d%Y_%H%M%S")

    return formatted_time


@cache
def get_output_path(
    output_path: str = "./", append_date: bool = True
) -> tuple[str, str]:
    """Get output path based on environment variable OUTPUT_PATH

    :param append_date: append a children folder with the date time, defaults to True
    :return: (output path, current time)
    """
    if "OUTPUT_PATH" in os.environ:
        output_path = os.environ["OUTPUT_PATH"]
        warnings.warn(
            f"OUTPUT_PATH is found in environment variables. Using path: {output_path}"
        )

    current_time = get_current_time()
    if append_date:
        output_path = os.path.join(output_path, current_time)

    return output_path, current_time
