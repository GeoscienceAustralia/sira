# from future.utils import itervalues
# these are required for defining the data model
from sifra.modelling.structural import (
    Element,
    Info,
    Base)

from sifra.modelling.component import Component

class IFSystem(Base):
    name = Element('str', "The model's name", 'model')
    description = Info('Represents a model (e.g. a "model of a powerstation")')

    components = Element('dict', 'The components that make up the infrastructure system', dict,
        [lambda x: [isinstance(y, Component) for y in x.itervalues()]])

    def add_component(self, name, component):
        self.components[name] = component

    def expose_to(self, intensity_param):
        return [c.expose_to(intensity_param) for c in self.components.itervalues()]


