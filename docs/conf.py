# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import glob
import importlib
import inspect
import pathlib
import re
import subprocess

import nntool

project = "🚂 NNTool"
copyright = "2025"
author = "Junhao Liu"
repository = "https://github.com/jhliu17/nntool"
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.linkcode",
    # "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
}
autodoc_inherit_docstrings = False
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"

autosummary_ignore_module_all = False


def linkcode_resolve(domain: str, info: dict) -> str | None:
    module = info.get("module", "")
    fullname = info.get("fullname", "")

    if not module or not fullname:
        return None

    try:
        objct = importlib.import_module(module)
        for name in fullname.split("."):
            objct = getattr(objct, name)

        while hasattr(objct, "__wrapped__"):
            objct = objct.__wrapped__

        file = inspect.getsourcefile(objct)
        if file is None:
            return None

        file_path = pathlib.Path(file).relative_to(root)
        lines, start = inspect.getsourcelines(objct)
        end = start + len(lines) - 1

        return f"{repository}/blob/{commit}/{file_path}#L{start}-L{end}"
    except Exception:
        return None


# add_function_parentheses = False
# default_role = "literal"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
# html_css_files = ["style/custom.css"]
# html_theme_options = {"collapse_navbar": False, "show_toc_level": 2}
templates_path = ["_templates"]
html_css_files = [
    "custom.css",
]
html_show_sourcelink = False
html_sourcelink_suffix = ""
html_theme_options = {
    "sidebar_hide_name": True,
    "top_of_page_buttons": ["view"],
}
