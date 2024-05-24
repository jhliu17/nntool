def get_current_time() -> str: ...
def get_output_path(output_path: str = ..., append_date: bool = ...) -> tuple[str, str]:
    """Get output path based on environment variable OUTPUT_PATH

    :param append_date: append a children folder with the date time, defaults to True
    :return: output path
    """
    ...
