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
add_function_parentheses = False
default_role = "literal"

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


def linkcode_resolve(domain: str, info: dict) -> str:
    module = info.get("module", "")
    fullname = info.get("fullname", "")

    if not module or not fullname:
        return None

    objct = importlib.import_module(module)
    for name in fullname.split("."):
        objct = getattr(objct, name)

    while hasattr(objct, "__wrapped__"):
        objct = objct.__wrapped__

    try:
        file = inspect.getsourcefile(objct)
        file = pathlib.Path(file).relative_to(root)

        lines, start = inspect.getsourcelines(objct)
        end = start + len(lines) - 1
    except Exception:
        return None
    else:
        return f"{repository}/blob/{commit}/{file}#L{start}-L{end}"


## Edit HTML


def edit_html(app, exception):
    if exception:
        raise exception

    for file in glob.glob(f"{app.outdir}/**/*.html", recursive=True):
        with open(file, "r") as f:
            text = f.read()

        # fmt: off
        text = text.replace('<a class="muted-link" href="https://pradyunsg.me">@pradyunsg</a>\'s', '')
        text = text.replace('<span class="pre">[source]</span>', '<i class="fa-solid fa-code"></i>')
        text = re.sub(r'(<a class="reference external".*</a>)(<a class="headerlink".*</a>)', r'\2\1', text)
        # fmt: on

        with open(file, "w") as f:
            f.write(text)


def setup(app):
    app.connect("build-finished", edit_html)
