import sys
import abc
import json
from sifra.modelling.component_db import (
    _Session,
    ModelTable,
    Attribute)


class SerialisationProvider(object):
    """
    Provides access to an object that can be used to serialise models or other
    components.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_db(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_db(self):
        raise NotImplementedError()


def _addAttributes(document, attributes):
    if attributes is not None:
        for name, value in attributes.iteritems():
            document.attributes.append(
                Attribute(name=name, value=value))


def _attributesToDict(attributes):
    return {a.name: a.value for a in attributes}


class SqliteDBProxy(object):
    def get(self, obj_id):
        session = _Session()
        instance = session.query(Model).filter(Model.id == obj_id).one()
        res = json.loads(instance.json_doc)
        res['_attributes'] = _attributesToDict(instance.attributes)
        session.close()
        return res


class SqliteSerialisationProvider(SerialisationProvider):
    def get_db(self):
        return SqliteDBProxy()

    def delete_db(self):
        session = _Session()
        session.query(ModelTable).delete()
        session.query(Attribute).delete()
        session.commit()
