import sys
import abc
import json
import couchdb
from sifra.components import (
    _Session,
    Component,
    ComponentAttribute,
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



def _addAttributes(component, attributes):
    if attributes is not None:
        for name, value in attributes.iteritems():
            component.attributes.append(
                ComponentAttribute(name=name, value=value))



def _attributesToDict(attributes):
    return {a.name: a.value for a in attributes}



class CouchDBProxy(object):
    def __init__(self, db):
        self._db = db

    def get(self, obj_id):
        return self._db.get(obj_id)

    def save(self, obj, category=None, attributes=None):
        clazz = obj['class']
        oid, over = self._db.save(obj)
        if category is not None:
            category = getComponentCategory(**category)
        component = Component(category=category, clazz=clazz)
        session = _Session()
        session.add(component)
        _addAttributes(component, attributes)
        session.commit()
        return [oid, over]



class CouchSerialisationProvider(SerialisationProvider):
    """
    Implementation of :py:class:`SerialisationProvider` for
    `CouchDB <http://couchdb.apache.org/>`_.
    """

    _all_provider_instances = []

    def __init__(self, server_url, db_name):
        self._server_url = server_url
        self._server = couchdb.Server(server_url)
        self._db_name = db_name
        self._db = None
        CouchSerialisationProvider._all_provider_instances.append(self)

    def _connect(self):
        try:
            # note that this causes an error in the couch db server... but that
            # is the way the python-couchdb library is designed.
            self._db = self._server[self._db_name]
        except couchdb.http.ResourceNotFound:
            self._db = self._server.create(self._db_name)

    def get_db(self):
        # The following is not thread safe, but I don't think that creating
        # multiple connections will cause problems... so don't worry about it.
        if self._db is None:
            self._connect()
        return CouchDBProxy(self._db)

    def delete_db(self):
        if self._db is not None:
            self._server.delete(self._db_name)
            for prov in CouchSerialisationProvider._all_provider_instances:
                if prov._server_url == self._server_url and prov._db_name == self._db_name:
                    prov._db = None



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
            component = Component(
                category=category,
                json_doc=json.dumps(obj),
                clazz=clazz)
            session.add(component)
            # call flush here to get the component's id (the default name)
            session.flush()
            _addAttributes(component, attributes)
            session.commit()
            return [component.id, None]
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

