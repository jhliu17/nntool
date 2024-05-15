import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


def list_pyx_files(directory):
    pyx_files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".pyx") or filename.endswith(".py"):
                pyx_files.append(os.path.join(dirpath, filename))
    return pyx_files


pyx_files = list_pyx_files("nntool")

extensions = [
    Extension(pyx.replace(".pyx", "").replace("/", "."), [pyx]) for pyx in pyx_files
]

setup(
    name="nntool",
    version="0.2.3",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
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
        "toml>=0.10",
    ],
    extras_require={"dev": ["pytest>=8.0.2", "jax[cpu]>=0.4.0", "torch>=2.2.0"]},
    setup_requires=["cython"],
    # Additional metadata
    author="Junhao Liu",
    author_email="junhaoliu17@gmail.com",
    description="neural network tool for research",
    license="MIT",
    keywords="deep learning, neural network, research",
    url="https://github.com/jhliu17/nntool",  # Project home page
)
