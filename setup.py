import setuptools
from sifra.__about__ import (
    __packagename__,
    __description__,
    __url__,
    __version__,
    __author__,
    __email__,
    __license__,
    __copyright__
)


if __name__ == '__main__':
    setuptools.setup(
        name=__packagename__,
        version=__version__,
        description=__description__,
        url=__url__,
        author=__author__,
        author_email = __email__,
        license=__license__,
        test_suite='.tests',
        tests_require=['nose'],
        install_requires=['graphviz'],
        packages=setuptools.find_packages(),
        package_dir={__packagename__: __packagename__}
    )
