from sifra.structural import (
    CouchSerialisationProvider,
    Element,
    generate_element_base)

COUCH_URL = 'http://couch:5984'
DB_NAME = 'models'
provider = CouchSerialisationProvider(COUCH_URL, DB_NAME)
Base = generate_element_base(provider)

class Unreachable_test_util(object): pass

class ResponseModel(Base):
    unreachable = Element('Unreachable_test_util', 'Cannot be seen in test mod', Unreachable_test_util)

    def __call__(self, pga):
        raise NotImplementedError()

