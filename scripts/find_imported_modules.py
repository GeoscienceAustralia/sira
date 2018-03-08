import os

python_file_paths = []
for root, dir_names, file_names in os.walk(os.getcwd()):
    for file_name in file_names:
        if file_name.endswith('.py'):
            python_file_paths.append(os.path.join(root, file_name))

packages=[]
for python_file_path in python_file_paths:
    with open(python_file_path, 'r') as infile:
        for line in infile:
            line = line.strip()

            if len(line)>2:
                if 'sifra' not in line and line[0] != '#':
                    if 'import' in line:
                        if 'import' in line[0:6]:
                            line=line.replace('import','').strip()
                            if 'as' in line:
                                line=line.split('as')[0].strip()
                                line = line.strip()
                                if line not in packages:
                                    packages.append(line)
                            else:
                                line = line.strip()

                                if line not in packages:
                                    packages.append(line)
                        else:
                            if '__future__' not in line:
                                if 'from' in line:
                                    line = line.replace('from','').strip()
                                    line = line .split('import')[0]

                                    if '.' in line:
                                        line = line.split('.')[0]
                                    line = line.strip()
                                    if line not in packages:
                                        packages.append(line)
for pkg in packages:
    print(pkg)

"""
setuptools
itertools
os
ast
openpyxl
argparse
logging
sys
shlex
sphinx_rtd_theme
collections
docutils
docutils.parsers.rst.directives
sphinx
codecs
re
sets
pybtex
latex_codec
json
pprint
pand
time
matplotlib.pyplot
matplotlib
numpy
scipy
lmfit
warnings
brewer2mpl
colorama
datetime
pickle
zipfile
parmap
seaborn
model_ingest
copy
scipy.stats
igraph
sys, os
inspect
networkx
lib
functools
flask
flask_cors
flask_wtf
unittest
netCDF4

"""