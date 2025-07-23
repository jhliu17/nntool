import subprocess
from rich.console import Console
from rich.table import Table


def nvidia_smi_gpu_memory_stats() -> dict:
    """
    Parse the nvidia-smi output and extract the memory used stats.
    """
    out_dict = {}
    try:
        sp = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        out_str = sp.communicate()
        out_list = out_str[0].decode("utf-8").split("\n")
        out_dict = {}
        for item in out_list:
            if " MiB" in item:
                gpu_idx, mem_used = item.split(",")
                gpu_key = f"gpu_{gpu_idx}_mem_used_gb"
                out_dict[gpu_key] = int(mem_used.strip().split(" ")[0]) / 1024
    except FileNotFoundError:
        raise Exception("Failed to find the 'nvidia-smi' executable for printing GPU stats")
    except subprocess.CalledProcessError as e:
        raise Exception(f"nvidia-smi returned non zero error code: {e.returncode}")

    return out_dict


def nvidia_smi_gpu_memory_stats_str() -> str:
    """
    Parse the nvidia-smi output and extract the memory used stats.
    Returns a rich-formatted table string for pretty printing.
    """
    stats = nvidia_smi_gpu_memory_stats()

    # Create a Rich table
    table = Table(title="GPU Memory Usage", show_header=True, header_style="bold magenta")
    table.add_column("GPU", style="cyan", width=8)
    table.add_column("Memory Used (GB)", style="green", justify="right", width=18)

    # Add rows for each GPU
    for key, value in stats.items():
        gpu_id = key.replace("gpu_", "").replace("_mem_used_gb", "")
        table.add_row(f"GPU {gpu_id}", f"{value:.4f}")

    # Create console and capture the table as string
    console = Console(force_terminal=True, width=40)
    with console.capture() as capture:
        console.print(table)

    return capture.get()


def print_nvidia_smi_gpu_memory_stats() -> None:
    """
    Print GPU memory stats using Rich formatting directly to the console.
    """
    stats = nvidia_smi_gpu_memory_stats()

    # Create a Rich table
    table = Table(title="GPU Memory Usage", show_header=True, header_style="bold magenta")
    table.add_column("GPU", style="cyan", width=8)
    table.add_column("Memory Used (GB)", style="green", justify="right", width=18)

    # Add rows for each GPU
    for key, value in stats.items():
        gpu_id = key.replace("gpu_", "").replace("_mem_used_gb", "")
        table.add_row(f"GPU {gpu_id}", f"{value:.4f}")

    # Print directly to console
    console = Console()
    console.print(table)
