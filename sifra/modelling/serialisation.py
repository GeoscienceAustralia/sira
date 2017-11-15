import sys
import abc
import json
from sifra.modelling.component_db import (
    _Session,
    Document,
    Component,
    DocumentAttribute,
    getComponentCategory)



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
                DocumentAttribute(name=name, value=value))


def _attributesToDict(attributes):
    return {a.name: a.value for a in attributes}


class SqliteDBProxy(object):
    def get(self, obj_id):
        session = _Session()
        instance = session.query(Component).filter(Component.id == obj_id).one()
        res = json.loads(instance.json_doc)
        res['_attributes'] = _attributesToDict(instance.attributes)
        session.close()
        return res

    def save(self, obj, category=None, attributes=None):
        """
        :param obj: A json serialisable object.
        :param category: A dictionary of key value pairs suitable for passing to
            the constructor of :py:class:`sifra.components.ComponentCategory` as
            *\*\*kwargs*.
        """

        clazz = '.'.join(obj['class'])
        session = _Session()
        if category is not None:
            category = getComponentCategory(session, **category)
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


class SqliteSerialisationProvider(SerialisationProvider):
    def get_db(self):
        return SqliteDBProxy()

    def delete_db(self):
        session = _Session()
        session.query(Component).delete()
        session.commit()
