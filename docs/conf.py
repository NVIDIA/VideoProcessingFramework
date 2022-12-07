import os
import sys

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autosummary_imported_members = True

autodoc_default_options = {"members": True, "memer-order": "bysource"}

source_suffix = ".rst"
master_doc = "index"
project = "VPF"
copyright = "2022 NVIDIA Corporation"
author = ""

language = None

exclude_patterns = ["_build"]

pygments_style = "sphinx"

todo_include_todos = False

html_theme = "haiku"
htmlhelp_basename = "VPFdoc"
