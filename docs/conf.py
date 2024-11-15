import os
import sys

sys.path.insert(0, os.path.abspath("/Users/junhao/Working/code/nntool"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "🚂 NNTool"
copyright = "2024"
author = "Junhao Liu"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

add_module_names = False
autosummary_generate = True
autodoc_typehints = "none"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {"collapse_navbar": False, "show_toc_level": 2}
