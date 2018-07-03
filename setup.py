import setuptools
import sifra

PACKAGE_NAME = 'sifra'
DESCRIPTION = 'SIFRA: System for Infrastructure Facility Resilience Analysis'
URL = 'https://github.com/GeoscienceAustralia/sifra'
LICENSE = 'Apache v2.0'

MAINTAINER = 'Maruf Rahman'
CONTRIBUTORS = ['Maruf Rahman', 'Sudipta Basak', 'Sheece Gardezi']
MAINTAINER_EMAIL = 'maruf.rahman@ga.gov.au'

if __name__ == '__main__':
    setuptools.setup(
    name=PACKAGE_NAME,
    version=sifra.__version__,
    description=DESCRIPTION,
    url=URL,
    license=LICENSE,
    author=MAINTAINER,
    test_suite='.tests',
    tests_require=['nose'],
    install_requires=['graphviz'],
    packages=setuptools.find_packages(),
    package_dir={PACKAGE_NAME: PACKAGE_NAME}
    )
