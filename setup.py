from setuptools import setup, find_packages
from itertools import ifilter
from os import path
from ast import parse

if __name__ == '__main__':
    package_name = 'sifra'
    with open(path.join(package_name, '__init__.py')) as f:
        __version__ = parse(next(ifilter(lambda line: line.startswith('__version__'), f))).body[0].value.s

    setup(
        name=package_name,
        version=__version__,
        url='https://github.com/GeoscienceAustralia/sifra',
        license='LICENSE',
        author='Maruf Rahman, Sudipta Basak',
        test_suite='.tests',
        tests_require=['nose'],
        install_requires=['graphviz'],
        packages=find_packages(),
        package_dir={package_name: package_name}
    )
