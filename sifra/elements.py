import unittest as ut

# these are required for defining the data model
from sifra.structural import (
    CouchSerialisationProvider,
    Element,
    ValidationError,
    generate_element_base)



COUCH_URL = 'http://couch:5984'
DB_NAME = 'models'
provider = CouchSerialisationProvider(COUCH_URL, DB_NAME)
Base = generate_element_base(provider)



class ResponseModel(Base):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('__call__ is not impleme ted on {}'.format(
            self.__class__.__name__))

class Model(Base):
    components = Element('dict', 'A component', dict,
        [lambda x: [isinstance(y, Component) for y in x.itervalues()]])

    name = Element('str', "The model's name", 'model')

    def add_component(self, name, component):
        self.components[name] = component



class Component(Base):
    frag_func = Element('ResponseModel', 'A fragility function', Element.NO_DEFAULT)

    def expose_to(self, pga):
        return self.frag_func(pga)

