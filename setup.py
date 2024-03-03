from setuptools import setup, find_packages


setup(
    name="nntool",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your package dependencies here
        # e.g., 'requests>=2.0',
        "setuptools>=68.0.0",
        "submitit>=1.5.0",
        "tyro>=0.7.0",
        "matplotlib>=3.8.0",
        "jax[cpu]>=0.4.0",
        "seaborn>=0.13.2",
        "pytest>=8.0.2",
    ],
    # Additional metadata
    author="Junhao Liu",
    author_email="junhaoliu17@gmail.com",
    description="neural network tool for research",
    license="MIT",
    keywords="deep learning, neural network, research",
    url="https://github.com/jhliu17/nntool",  # Project home page
)
