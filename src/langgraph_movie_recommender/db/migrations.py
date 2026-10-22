import logging

from sqlalchemy import inspect

from langgraph_movie_recommender.db.models import Base

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


def _get_user_version(connection):
    return connection.exec_driver_sql("PRAGMA user_version").scalar_one()


def _set_user_version(connection, version):
    connection.exec_driver_sql(f"PRAGMA user_version = {int(version)}")


def _drop_indexes(connection, table_name):
    indexes = connection.exec_driver_sql(f'PRAGMA index_list("{table_name}")').all()

    for index in indexes:
        index_name = index[1]

        if index_name.startswith("sqlite_autoindex_"):
            continue

        connection.exec_driver_sql(f'DROP INDEX "{index_name}"')


def _rebuild_legacy_tables(connection, table_names):
    legacy_names = {}

    for table_name in table_names:
        _drop_indexes(connection, table_name)
        legacy_name = f"_legacy_{table_name}"
        connection.exec_driver_sql(f'ALTER TABLE "{table_name}" RENAME TO "{legacy_name}"')
        legacy_names[table_name] = legacy_name

    Base.metadata.create_all(connection)

    for table_name, legacy_name in legacy_names.items():
        columns = ", ".join(
            f'"{column.name}"' for column in Base.metadata.tables[table_name].columns
        )
        connection.exec_driver_sql(
            f'INSERT OR IGNORE INTO "{table_name}" ({columns}) '
            f'SELECT {columns} FROM "{legacy_name}"'
        )
        connection.exec_driver_sql(f'DROP TABLE "{legacy_name}"')


def _create_schema(connection):
    inspector = inspect(connection)
    managed_tables = set(Base.metadata.tables)
    existing_tables = set(inspector.get_table_names()) & managed_tables

    if existing_tables:
        _rebuild_legacy_tables(connection, sorted(existing_tables))
    else:
        Base.metadata.create_all(connection)


def _drop_profile_columns(connection):
    inspector = inspect(connection)
    existing_tables = set(inspector.get_table_names())
    legacy_tables = []

    for table_name in Base.metadata.tables:
        if table_name not in existing_tables:
            continue

        columns = {column["name"] for column in inspector.get_columns(table_name)}

        if "profile_id" in columns:
            legacy_tables.append(table_name)

    if legacy_tables:
        _rebuild_legacy_tables(connection, sorted(legacy_tables))


MIGRATIONS = {
    1: _create_schema,
    2: _drop_profile_columns,
}


def migrate(connection):
    current_version = _get_user_version(connection)

    if current_version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {current_version} is newer than supported "
            f"version {SCHEMA_VERSION}."
        )

    for version in range(current_version + 1, SCHEMA_VERSION + 1):
        logger.info("Applying database migration %d", version)
        MIGRATIONS[version](connection)
        _set_user_version(connection, version)
