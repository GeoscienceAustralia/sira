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

    def save(self):
        res = jsonify(self)

        clazz = '.'.join(obj['class'])
        session = _Session()

        try:
            document = Document(json_doc=json.dumps(obj))
            session.add(document)
            # call flush here to get the document's id (the default name)
            session.flush()
            _addAttributes(document, attributes)
            component = Component(
                category=category,
                document=document,
                clazz=clazz)
            session.add(component)
            session.commit()
            return component.id
        except Exception, e:
            session.rollback()
            raise e
        finally:
            session.close()
