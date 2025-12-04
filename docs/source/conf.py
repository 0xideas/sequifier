# pyright: ignore
# ruff: noqa
# fmt: off
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))
# -- Project information -----------------------------------------------------
# https://smv_copy_srcdirs.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sequifier'
copyright = '2025, Leon Luithlen'
author = 'Leon Luithlen'
release = 'v1.0.0.3'
html_baseurl = 'https://www.sequifier.com/'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_multiversion'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']


napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True


# Defines how versions are sorted in the version switcher.
smv_tag_whitelist = r'^v\d+\.\d+\.\d+\.\d+$'    # Tags like v1.2.3.4
smv_branch_whitelist = r'^main$'          # Only include the main branch
smv_remote_whitelist = r'^origin$'        # Only use the 'origin' remote
smv_released_pattern = r'^refs/tags/v\d+\.\d+\.\d+\.\d+$' # Match 4-part tags
smv_outputdir_format = '{ref.name}'       # Use branch/tag name for output directory
smv_prefer_remote_refs = False            # Use local refs first if they exist
smv_copy_srcdirs = ['../../src']

# Optional: Add a banner to warn users viewing non-released versions
smv_warning_banner_enabled = True
smv_show_banner_if = r'^(?!v\d+\.\d+\.\d+\.\d+$).*$' # Show banner on anything not like vX.Y.Z.A

# Custom sidebar configuration to include the version switcher
html_sidebars = {
    '**': [
        'sidebar/brand.html',         # From Furo theme
        'sidebar/search.html',        # From Furo theme
        'sidebar/scroll-start.html',  # From Furo theme
        'sidebar/navigation.html',    # From Furo theme (toctree)
        'sidebar/versions.html',      # Our custom version switcher template
        'sidebar/scroll-end.html',    # From Furo theme
    ],
}
