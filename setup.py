import os
from pathlib import Path
from typing import List, Tuple

from Cython.Build import cythonize
from setuptools import setup, find_packages, Command, Extension
from cythonpackage.build import _build_py


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


class custom_build_py(_build_py):
    def find_package_modules(
        self, package: str, package_dir: str
    ) -> List[Tuple[str, str, str]]:
        """Remove source code"""
        modules: List[Tuple[str, str, str]] = super().find_package_modules(
            package, package_dir
        )
        if self.remove_source:
            filtered_modules = []
            for pkg, mod, filepath in modules:
                _path = Path(filepath)
                if (
                    _path.suffix in [".py", ".pyx"]
                    and ("__init__.py" != _path.name)
                    # and filepath not in self._exclude ) :
                    and Path(_path.parent, "__compile__.py").exists()
                ):
                    continue
                filtered_modules.append(
                    (
                        pkg,
                        mod,
                        filepath,
                    )
                )
            return filtered_modules
        else:
            return modules


package_info = dict(
    name="nntool",
    version="1.1.0",
    python_requires=">=3.9",
    install_requires=[
        # List your package dependencies here
        # e.g., 'requests>=2.0',
        "setuptools>=68.0.0",
        "submitit>=1.5.0",
        "tyro>=0.8.12",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.2",
        "wandb>=0.15.0",
        "tomli>=2.0.1",
        "cythonpackage",
    ],
    cmdclass={
        "clean_cython": CleanCython,
    },
    extras_require={
        "dev": [
            "pytest>=8.0.2",
            "jax[cpu]>=0.4.0",
            "torch>=2.2.0",
            "mypy",
            "accelerate",
        ]
    },
    # Additional metadata
    author="Junhao Liu",
    author_email="junhaoliu17@gmail.com",
    description="neural network tool for research",
    license="MIT",
    keywords="deep learning, neural network, research",
    url="https://github.com/jhliu17/nntool",  # Project home page
)

if os.getenv("NNTOOL_PYTHON_BUILD"):
    setup(
        packages=find_packages(exclude=["tests"]),
        exclude_package_data={
            "": ["*.pyi"]
        },  # for source release, ignore all pyi files
        **package_info,
    )
else:
    # Specify the Cython modules to build
    cython_extensions = [
        Extension(
            name="nntool.slurm.csrc.__compile__",
            sources=["nntool/slurm/csrc/**.py"],
        ),
        Extension(
            name="nntool.plot.csrc.__compile__",
            sources=["nntool/plot/csrc/**.py"],
        ),
    ]
    package_info["cmdclass"]["build_py"] = custom_build_py
    setup(
        packages=find_packages(exclude=["tests"]),
        # cythonpackage={
        #     "inject_ext_modules": False,
        #     "inject_init": False,
        #     "remove_source": True,
        #     "compile_py": True,
        #     "optimize": 1,
        #     "exclude": ["tests/*"],  # List of glob
        # },
        setup_requires=["cython"],
        ext_modules=cythonize(
            cython_extensions,
            build_dir="build/cythonpackage",
            compiler_directives={"language_level": 3},
        ),
        **package_info,
    )
