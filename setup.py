from setuptools import setup, find_packages
from itertools import ifilter
from os import path
from ast import parse

DESCRIPTION = 'SIFRA: System for Infrastructure Facility Resilience Analysis'
MAINTAINER = 'Maruf Rahman'
CONTRIBUTORS = ['Maruf Rahman', 'Sudipta Basak']
MAINTAINER_EMAIL = 'maruf.rahman@ga.gov.au'
URL = 'https://github.com/GeoscienceAustralia/sifra'
LICENSE = 'Apache v2.0'

if __name__ == '__main__':
    package_name = 'sifra'
    with open(path.join(package_name, '__init__.py')) as f:
        __version__ = parse(next(ifilter(lambda line: line.startswith('__version__'), f))).body[0].value.s

    setup(
        name=package_name,
        version=__version__,
        url=URL,
        license=LICENSE,
        author=MAINTAINER,
        test_suite='.tests',
        tests_require=['nose'],
        install_requires=['graphviz'],
        packages=find_packages(),
        package_dir={package_name: package_name}
    )
