# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "SynthTIGER"
copyright = "2021, Clova AI Research, NAVER Corp."
author = "Clova AI Research, NAVER Corp."


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Autodoc configuration ---------------------------------------------------

autodoc_member_order = "bysource"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-api-keyword": "#cf222e",
        "color-api-pre-name": "#24292f",
        "color-api-name": "#8250df",
        "color-api-overall": "#24292f",
        "color-link": "#24292f",
    },
    "dark_css_variables": {
        "color-api-keyword": "#ff7b72",
        "color-api-pre-name": "#c9d1d9",
        "color-api-name": "#d2a8ff",
        "color-api-overall": "#c9d1d9",
        "color-link": "#c9d1d9",
    },
}
html_title = project

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
