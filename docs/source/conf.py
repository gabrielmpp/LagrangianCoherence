import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',]

project = 'Lagrangian Coherence'
copyright = '2019, Gabriel Perez'
author = 'Gabriel Perez'
version = ''
release = '0.9'
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'
html_theme = 'alabaster'
html_static_path = ['_static']
