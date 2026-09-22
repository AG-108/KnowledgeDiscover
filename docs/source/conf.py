# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Resolve the repository root explicitly for autodoc imports.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# -- Project information -----------------------------------------------------

project = "KD"
copyright = "2025, Scientific-Artificial-Intelligence-Lab"
author = "Scientific-Artificial-Intelligence-Lab"

# The full version, including alpha/beta/rc tags
release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",  # Enable rendered mathematical expressions.
    "sphinx.ext.intersphinx",  # Enable cross-document references.
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Build the primary documentation in Chinese.
language = "en"
locale_dirs = ["locale/"]  # Store gettext catalogs beside the documentation sources.
gettext_compact = False  # Keep gettext catalogs uncompressed for easier inspection.
languages = ["en", "zh_CN"]  # Declare the translations produced by the documentation build.

# Include class and function members in generated API pages.
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"  # Use the standard Alabaster theme.

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Expose language-switch links in the theme sidebar.
html_theme_options = {
    "github_user": "Scientific-Artificial-Intelligence-Lab",
    "github_repo": "kd",
    "description": "Knowledge Discovery Documentation",
    "fixed_sidebar": True,
    "show_powered_by": False,  # Hide the default Sphinx footer.
    "github_banner": True,  # Link the theme to the GitHub repository.
    "github_button": True,  # Show the GitHub star and fork buttons.
}
