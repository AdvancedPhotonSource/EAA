"""Sphinx configuration for the EAA documentation."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


project = "Experiment Automation Agents"
author = "EAA contributors"
copyright = "2026, EAA contributors"
release = "latest"

extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
root_doc = "index"
autosectionlabel_prefix_document = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "pydata_sphinx_theme"
html_title = "EAA Documentation"
html_static_path = []
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc"],
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navigation_with_keys": True,
}
html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html"],
}
