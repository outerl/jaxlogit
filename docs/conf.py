# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "jaxlogit"
copyright = "2025, Jan Zill, Tyler Pearn, Evelyn Bond"
author = "Jan Zill, Tyler Pearn, Evelyn Bond"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser",
    "nbsphinx",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

nbsphinx_custom_formats = {
    '.py': ['jupytext.reads', {'fmt': 'py:percent'}],
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {"autosummary": True}
autosummary_generate = True



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
