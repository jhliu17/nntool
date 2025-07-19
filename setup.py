import os
from setuptools import setup, find_packages, Command, Extension


class CleanCython(Command):
    description = "custom clean command to remove Cython build artifacts"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for dirpath, _, filenames in os.walk("."):
            for filename in filenames:
                if filename.endswith((".c", ".so")):
                    file_path = os.path.join(dirpath, filename)
                    print(f"Removing file: {file_path}")
                    os.remove(file_path)


def get_csrc_files(folder: str) -> list[str]:
    source_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".py") and not file.endswith("__init__.py"):
                source_files.append(os.path.join(root, file))
    return source_files


package_info = dict(
    cmdclass={
        "clean_cython": CleanCython,
    },
)

if os.getenv("NNTOOL_PYTHON_BUILD"):
    setup(
        packages=find_packages(exclude=["tests"]),
        exclude_package_data={"": ["*.pyi"]},  # for source release, ignore all pyi files
        **package_info,
    )
else:
    from Cython.Build import cythonize

    # Specify the Cython modules to build
    cython_extensions = [
        Extension(
            name="nntool.slurm.core.__compile__",
            sources=get_csrc_files("nntool/slurm/core/"),
        ),
        Extension(
            name="nntool.plot.core.__compile__",
            sources=get_csrc_files("nntool/plot/core/"),
        ),
    ]

    setup(
        packages=find_packages(exclude=["tests"]),
        cythonpackage={
            "inject_ext_modules": False,
            "inject_init": True,
            "remove_source": True,
            "compile_py": True,
            "optimize": 1,
            "exclude": ["tests/*"],  # List of glob
        },
        ext_modules=cythonize(
            cython_extensions,
            build_dir="build/cythonpackage",
            compiler_directives={"language_level": 3},
        ),
        **package_info,
    )
