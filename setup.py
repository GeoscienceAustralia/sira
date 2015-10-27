try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'A System for Infrastructure Earthquake Risk Analysis',
    'author': 'Maruf Rahman',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'maruf.rahman@ga.gov.au',
    'version': '0.1.0',
    'install_requires': ['nose'],
    'packages': ['NAME'],
    'scripts': [],
    'name': 'sira'
}

setup(**config)