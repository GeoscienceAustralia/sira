from setuptools import setup, find_packages
from itertools import ifilter
from os import path
from ast import parse

if __name__ == '__main__':
    package_name = 'sira'
    with open(path.join(package_name, '__init__.py')) as f:
        __version__ = parse(next(ifilter(lambda line: line.startswith('__version__'), f))).body[0].value.s

    setup(
        name=package_name,
        author='Sudipta Basak, Maruf Rahman',
        version=__version__,
        test_suite='.tests',
        packages=find_packages(),
        package_dir={package_name: package_name}
    )
