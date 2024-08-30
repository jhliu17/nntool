import sys
import time
import subprocess
from nntool.utils import get_output_path


def test_get_output_path(tmp_path):
    output_path = get_output_path("tmp_path", append_date=True)
    time.sleep(5)
    result = subprocess.run(
        ["python", "-m", "tests.test_utils", tmp_path], capture_output=True, text=True
    )
    assert result.stdout.strip() == output_path[0]


if __name__ == "__main__":
    output_path, current_time = get_output_path(sys.argv[1], append_date=True)
    print(output_path)
