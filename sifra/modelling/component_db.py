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

_Base = declarative_base()


class Attribute(_Base):
    __tablename__ = "attributes"
    document_id = Column(Integer, primary_key=True)
    name = Column(String(32), primary_key=True)
    value = Column(String())


class Component(_Base):
    __tablename__ = "components"
    id = Column(Integer, primary_key=True)
    name = Column(String(), unique=True)
    clazz = Column(String())
    attributes = relationship('Attribute', cascade='all, delete-orphan')

    @property
    def json_doc(self):
        return self.document.json_doc

    @property
    def attributes(self):
        return self.document.attributes


class Model(_Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    name = Column(String(), unique=True)
    clazz = Column(String())

    components_json = Column(String())
    attributes = relationship('Attribute', cascade='all, delete-orphan')

    @property
    def attributes(self):
        return self.attributes


def getComponentCategories(hazard=None, sector=None, facility_type=None, component=None):
    # convert anything that evaluates to false to None
    hazard = hazard or None
    sector = sector or None
    facility_type = facility_type or None
    component = component or None

    session = _Session()
    rows = session.query(Attribute)

    if hazard is not None:
        rows = rows.filter(Attribute.name == hazard)
    if sector is not None:
        rows = rows.filter(Attribute.name == sector)
    if facility_type is not None:
        rows = rows.filter(Attribute.name == facility_type)
    if component is not None:
        rows = rows.filter(Attribute.name == component)

    return {
        'hazards': list(set((row.hazard for row in rows))),
        'sectors': list(set((row.sector for row in rows))),
        'facilities': list(set((row.facility_type for row in rows))),
        'components': list(set((row.component for row in rows)))}


def getComponentCategory(session, hazard, sector, facility_type, component):
    return session.query(Attribute) \
        .filter(Attribute.name == hazard) \
        .filter(Attribute.name == sector) \
        .filter(Attribute.name == facility_type) \
        .filter(Attribute.name == component).one()


def getAllInstances(cls):
    def getName(component):
        for a in component.attributes:
            if a.name == 'name':
                return a.value

    session = _Session()
    components = session.query(Component).filter(
        Component.clazz.in_(get_all_subclasses(cls)))
    return [{'name': getName(r), 'id': r.id} for r in components.all()]


def getInstance(instance_id):
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

