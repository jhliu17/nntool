def get_current_time() -> str:
    """get current time in this format: MMDDYYYY_HHMMSS

    :return: time in the format MMDDYYYY_HHMMSS
    """
    ...

def get_output_path(output_path: str = ..., append_date: bool = ...) -> tuple[str, str]:
    """Get output path based on environment variable OUTPUT_PATH.
    The output path is appended with the current time if append_date is True (e.g. /outputs/xxx/MMDDYYYY_HHMMSS).

    The result for the same input is cached.

    :param append_date: append a children folder with the date time, defaults to True
    :return: (output path, current time)
    """
    ...
