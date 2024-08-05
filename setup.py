import os
from setuptools import setup, find_packages, Command


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


package_info = dict(
    name="nntool",
    version="1.0.2rc1",
    python_requires=">=3.9",
    install_requires=[
        # List your package dependencies here
        # e.g., 'requests>=2.0',
        "setuptools>=68.0.0",
        "submitit>=1.5.0",
        "tyro>=0.7.0",
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
    setup(
        packages=find_packages(exclude=["tests"]),
        cythonpackage={
            "inject_ext_modules": True,
            "inject_init": False,
            "remove_source": True,
            "compile_py": True,
            "optimize": 1,
            "exclude": ["nntool/train/*.py", "tests/*"],  # List of glob
        },
        setup_requires=["cython", "cythonpackage[build]"],
        **package_info,
    )
