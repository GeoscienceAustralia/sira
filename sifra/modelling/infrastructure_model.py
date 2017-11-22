from sifra.modelling.structural import (
    Element,
    Info,
    Base)

from sifra.modelling.component import Component
from sifra.modelling.iodict import IODict


class Model(Base):

    name = Element('str', "The model's name", 'model')
    description = Info('Represents a model (e.g. a "model of a powerstation")')

    components = Element('IODict', 'The components that make up the infrastructure system', IODict)

    def add_component(self, name, component):
        """Add a component to the component dict"""
        self.components[name] = component

