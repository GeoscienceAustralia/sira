import os
import csv
import json
from sqlalchemy import Integer, String, create_engine, Column, ForeignKey
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.declarative import declarative_base
from sifra.settings import DB_CONNECTION_STRING, SQLITE_DB_FILE, BUILD_DB_TABLES
from sifra.modelling.utils import get_all_subclasses

from sifra.modelling.utils import jsonify, pythonify

import json

_Base = declarative_base()


class ModelTable(_Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    name = Column(String(), unique=True)
    clazz = Column(String())

    components_json = Column(String())
    attributes = relationship('Attribute',
                              primaryjoin="ModelTable.id==Attribute.model_id",
                              cascade='all, delete-orphan')


class Attribute(_Base):
    __tablename__ = "model_attributes"
    model_id = Column(Integer, ForeignKey('models.id'), primary_key=True)
    name = Column(String(100), primary_key=True)
    value = Column(String())


class Component(_Base):
    __tablename__ = "components"
    id = Column(Integer, primary_key=True)
    name = Column(String(), unique=True)
    clazz = Column(String())


def save_model(model):
    res = jsonify(model)

    clazz = '.'.join(res['class'])
    model_name = res['name']
    session = _Session()

    # Extract the attributes of the jsonified class
    model_attributes = dict()
    for var_name in res.keys():
        if var_name in model.__dict__ and is_attribute_type(getattr(model, var_name)):
            model_attributes[var_name] = res.pop(var_name)

    try:
        model_db = ModelTable(name=model_name,
                            clazz=clazz,
                            components_json=json.dumps(res))

        session.add(model_db)
        session.commit()

        # _addAttributes(res)
        for m_name, m_value in model_attributes.items():
            model_attribute = Attribute(model_id=model_db.id,
                                        name=m_name,
                                        value=m_value)
            session.add(model_attribute)

        session.commit()

        return model_db.id
    except Exception, e:
        session.rollback()
        raise e
    finally:
        session.close()


def load_model(obj_id):
    session = _Session()
    model_row = session.query(ModelTable).filter(ModelTable.id == obj_id).one()
    model_kwargs = json.loads(model_row.components_json)
    for ma in model_row.attributes:
        model_kwargs[ma.name] = ma.value

    res = pythonify(model_kwargs)
    session.close()
    return res


def is_attribute_type(var):
    isinstance(var, (str, int, float))


def getAllComponents(cls):
    def getName(component):
        for a in component.attributes:
            if a.name == 'name':
                return a.value

    session = _Session()
    components = session.query(Component).filter(
        Component.clazz.in_(get_all_subclasses(cls)))
    return [{'name': getName(r), 'id': r.id} for r in components.all()]


def getComponent(instance_id):
    session = _Session()
    return json.loads(session.query(Component).filter(
        Component.id == instance_id).one().json_doc)

_engine = create_engine(
    DB_CONNECTION_STRING,
    echo=False,
    connect_args={'check_same_thread': False},
    poolclass=StaticPool)
_Session = sessionmaker(bind=_engine)


if BUILD_DB_TABLES:
    _Base.metadata.create_all(_engine)
    with open(os.path.abspath(os.path.join(
        os.path.dirname(__file__), 'components.csv'))) as cin:
        _session = _Session()
        _reader = csv.DictReader(cin)
        _rows = {tuple(r.values()): r for r in _reader}
        for _row in _rows.itervalues():
            _session.add(Attribute(**_row))
        _session.commit()

