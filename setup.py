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
            if (
                file.endswith(".py")
                and not file.endswith("__init__.py")
                and not file.startswith(".")
            ):
                source_files.append(os.path.join(root, file))
    return source_files


exec(open("nntool/version.py").read())
package_info = dict(
    name="nntool",
    version=VERSION,  # noqa: F821 # type: ignore
    python_requires=">=3.9",
    install_requires=[
        # List your package dependencies here
        # e.g., 'requests>=2.0',
        "tyro",
        "setuptools>=68.0.0,<81.0.0",
        "submitit>=1.5.3",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.2",
        "wandb>=0.15.0",
        "tomli>=2.0.1",
        "Cython==3.0.12",
        "cythonpackage",
    ],
    cmdclass={
        "clean_cython": CleanCython,
    },
    extras_require={
        "dev": [
            "ruff",
            "accelerate",
            "pytest>=8.0.2",
            "jax[cpu]>=0.4.0",
            "torch>=2.2.0",
            "sphinx",
            "furo",
            "myst_parser",
            "sphinx-autodoc-typehints",
            "sphinx-copybutton",
        ]
    },
    # Additional metadata
    author="Junhao Liu",
    author_email="junhaoliu17@gmail.com",
    description="NNTool is a package built on top of submitit designed to provide simple abstractions to conduct experiments on Slurm for machine learning research.",
    license="MIT",
    keywords="deep learning, neural network, research",
    url="https://github.com/jhliu17/nntool",  # Project home page
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
            name="nntool.slurm.csrc.__compile__",
            sources=get_csrc_files("nntool/slurm/csrc/"),
        ),
        Extension(
            name="nntool.plot.csrc.__compile__",
            sources=get_csrc_files("nntool/plot/csrc/"),
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
