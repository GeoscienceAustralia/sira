import os

SQLITE_DB_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'db', 'sqlite.db'))

BUILD_DB_TABLES = not os.path.exists(SQLITE_DB_FILE)

DB_CONNECTION_STRING = 'sqlite:///{}'.format(SQLITE_DB_FILE)

