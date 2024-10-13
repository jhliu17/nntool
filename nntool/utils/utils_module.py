import os
import datetime
import tomli


def get_current_time() -> str:
    """get current time in this format: MMDDYYYY/HHMMSS

    :return: time in the format MMDDYYYY/HHMMSS
    """
    # Get the current time
    current_time = datetime.datetime.now()

    # Format the time (MDY/HMS)
    formatted_time = current_time.strftime("%m%d%Y/%H%M%S")

    return formatted_time


def read_toml_file(file_path: str) -> dict:
    """Read a toml file and return the content as a dictionary

    :param file_path: path to the toml file
    :return: content of the toml file as a dictionary
    """
    with open(file_path, "rb") as f:
        content = tomli.load(f)

    return content


def get_output_path(
    output_path: str = "./",
    append_date: bool = True,
    cache_into_env: bool = True,
) -> tuple[str, str]:
    """Get output path based on environment variable OUTPUT_PATH and NNTOOL_OUTPUT_PATH.
    The output path is appended with the current time if append_date is True (e.g. /OUTPUT_PATH/xxx/MMDDYYYY/HHMMSS).

    :param append_date: append a children folder with the date time, defaults to True
    :param cache_into_env: whether cache the newly created path into env, defaults to True
    :return: (output path, current time)
    """
    if "OUTPUT_PATH" in os.environ:
        output_path = os.environ["OUTPUT_PATH"]
        current_time = "" if not append_date else get_current_time()
    elif "NNTOOL_OUTPUT_PATH" in os.environ:
        # reuse the NNTOOL_OUTPUT_PATH if it is set
        output_path = os.environ["NNTOOL_OUTPUT_PATH"]
        current_time = "" if not append_date else os.environ["NNTOOL_OUTPUT_PATH_DATE"]
    else:
        current_time = get_current_time()
        if append_date:
            output_path = os.path.join(output_path, current_time)

        if cache_into_env:
            os.environ["NNTOOL_OUTPUT_PATH"] = output_path
            os.environ["NNTOOL_OUTPUT_PATH_DATE"] = current_time
            print(
                f"OUTPUT_PATH is not found in environment variables. NNTOOL_OUTPUT_PATH is set using path: {output_path}"
            )

    return output_path, current_time
