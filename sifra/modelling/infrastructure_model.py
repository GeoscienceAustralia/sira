from sifra.modelling.structural import (
    Element,
    Info,
    Base)

from sifra.modelling.component import Component


class Model(Base):
    description = Info('Represents a model (e.g. a "model of a powerstation")')

    components = Element('dict', 'A component', dict,
        [lambda x: [isinstance(y, Component) for y in x.itervalues()]])

    name = Element('str', "The model's name", 'model')

    def add_component(self, name, component):
        self.components[name] = component
