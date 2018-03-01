import os
from os.path import exists

from sifra.sifraclasses import _readfile


CONF_FILENAME = os.path.join(os.path.dirname(__file__),
                             'C:\\Users\\u12089\\Desktop\\sifra-dev\\tests\\test_simple_series_struct_dep.conf')


if __name__ == "__main__":
    # unittest.main()
    conf = _readfile(CONF_FILENAME)

    print(type(_readfile(CONF_FILENAME)['TIME_UNIT'])==list)
