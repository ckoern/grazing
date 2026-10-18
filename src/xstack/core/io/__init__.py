"""Import and export of data files.

Measured curves are read with `read_curves` and structure models with `read_models`; both
choose the file handler by suffix from registries that plugins can extend.
"""

from xstack.core.io.data import DataParser, data_parsers, parser_for, read_curves
from xstack.core.io.delimited import UNITS, CsvLayout, CsvParser, Quantity
from xstack.core.io.models import (
    ImportedModel,
    ModelReader,
    model_readers,
    read_models,
    reader_for,
)
from xstack.core.io.reflectivity import (
    REFLECTIVITY_COLUMNS,
    reflectivity_table,
    write_reflectivity_csv,
)

__all__ = [
    "REFLECTIVITY_COLUMNS",
    "UNITS",
    "CsvLayout",
    "CsvParser",
    "DataParser",
    "ImportedModel",
    "ModelReader",
    "Quantity",
    "data_parsers",
    "model_readers",
    "parser_for",
    "read_curves",
    "read_models",
    "reader_for",
    "reflectivity_table",
    "write_reflectivity_csv",
]
