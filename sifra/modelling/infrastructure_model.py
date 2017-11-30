from sifra.modelling.structural import (
    Element,
    Info,
    Base)

from sifra.modelling.iodict import IODict
from sifra.modelling.utils import jsonify, pythonify

from sifra.modelling.component_db import (
    _Session,
    Model as ModelRow)


class Model(Base):

    name = Element('str', "The model's name", 'model')
    description = Info('Represents a model (e.g. a "model of a powerstation")')

    components = Element('IODict', 'The components that make up the infrastructure system', IODict)

    def add_component(self, name, component):
        """Add a component to the component dict"""
        self.components[name] = component

    def save(self):
        res = jsonify(self)

        clazz = '.'.join(res['class'])
        model_name = res['name']
        session = _Session()

        try:
            # _addAttributes(res)
            model_db = ModelRow(
                name=model_name,
                clazz=clazz,
                components_json=str(res)
            )
            session.add(model_db)
            session.commit()
            return model_db.id
        except Exception, e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get(self, obj_id):
        session = _Session()
        instance = session.query(Model).filter(Model.id == obj_id).one()
        res = pythonify(instance.json_doc)
        session.close()
        return res
