import shutil


def is_latex_available():
    """Check if LATEX is available on the system by looking for latex command"""
    return shutil.which("latex") is not None


def is_slurm_available():
    """Check if SLURM is available on the system by looking for srun command"""
    return shutil.which("srun") is not None
