"""Centralised JSON loading and validation for every benchmark specification file.

All schema-version gating, field presence, type, range, and enum checking happens here.
No other module may call :func:`json.load` on a specification file, so that validation
policy has exactly one home.

Validation is deliberately **strict**: an unknown key is an error, not something to
ignore. A misspelled ``deadline_sec`` must never be silently read as "field absent, use
the default" -- that would turn a typo into a different benchmark.

Migrations
----------
V1 supports ``schema_version`` ``"1.0"`` and nothing else. When a future version needs to
read older files, the migration step belongs in :func:`_migrate`, called from
:func:`open_document` after parsing and before field validation, so that every loader
gains migration support at once. It is documented, not implemented, in V1.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, TypeVar

from missionbench import SCHEMA_VERSION

__all__ = [
    "SUPPORTED_SCHEMA_VERSIONS",
    "DocumentReader",
    "SchemaValidationError",
    "SchemaVersionError",
    "open_document",
    "read_json_object",
]

#: Every schema version this release can load. Kept as a set so that a future release
#: supporting both "1.0" and "1.1" needs no structural change here.
SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset({SCHEMA_VERSION})

_SCHEMA_VERSION_KEY: Final = "schema_version"

_EnumT = TypeVar("_EnumT", bound=StrEnum)

#: JSON scalar types permitted inside open-ended ``parameters`` maps.
_SCALAR_TYPES: Final = (str, int, float, bool, type(None))


class SchemaValidationError(ValueError):
    """A specification file does not conform to its schema.

    Carries the offending location in its message so the failure is actionable without
    a debugger.
    """


class SchemaVersionError(SchemaValidationError):
    """A specification file declares a ``schema_version`` this release cannot load."""


class DocumentReader:
    """Validating accessor over one JSON object.

    Rejects unknown keys on construction, then type- and range-checks each field as it is
    read. ``context`` is a human-readable location such as
    ``"contract_001.json -> Contract"`` and is prefixed to every error message.
    """

    __slots__ = ("_context", "_data")

    def __init__(
        self,
        data: Mapping[str, Any],
        *,
        context: str,
        allowed_fields: Iterable[str],
    ) -> None:
        self._data = data
        self._context = context
        unknown = sorted(set(data) - set(allowed_fields))
        if unknown:
            raise SchemaValidationError(
                f"{context}: unknown field(s) {unknown}. "
                f"Allowed fields are {sorted(allowed_fields)}. "
                "Unknown fields are rejected so that a typo is never read as a default."
            )

    @property
    def context(self) -> str:
        return self._context

    # -- primitives ---------------------------------------------------------------

    def _require(self, key: str) -> Any:
        if key not in self._data:
            raise SchemaValidationError(f"{self._context}: missing required field {key!r}")
        return self._data[key]

    def get_str(self, key: str, *, allow_empty: bool = False) -> str:
        value = self._require(key)
        if not isinstance(value, str):
            raise SchemaValidationError(
                f"{self._context}: field {key!r} must be a string, got {type(value).__name__}"
            )
        if not allow_empty and not value.strip():
            raise SchemaValidationError(f"{self._context}: field {key!r} must not be empty")
        return value

    def get_optional_str(self, key: str) -> str | None:
        """Read a nullable string. A missing key and an explicit ``null`` both yield ``None``."""
        if self._data.get(key) is None:
            return None
        return self.get_str(key)

    def get_int(self, key: str, *, minimum: int | None = None) -> int:
        value = self._require(key)
        # bool is a subclass of int; accepting it here would silently turn true into 1.
        if not isinstance(value, int) or isinstance(value, bool):
            raise SchemaValidationError(
                f"{self._context}: field {key!r} must be an integer, got {type(value).__name__}"
            )
        if minimum is not None and value < minimum:
            raise SchemaValidationError(
                f"{self._context}: field {key!r} must be >= {minimum}, got {value}"
            )
        return value

    def get_float(
        self,
        key: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive_minimum: float | None = None,
    ) -> float:
        value = self._require(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaValidationError(
                f"{self._context}: field {key!r} must be a number, got {type(value).__name__}"
            )
        number = float(value)
        if number != number:  # NaN: every comparison below would pass vacuously.
            raise SchemaValidationError(f"{self._context}: field {key!r} must not be NaN")
        if minimum is not None and number < minimum:
            raise SchemaValidationError(
                f"{self._context}: field {key!r} must be >= {minimum}, got {number}"
            )
        if exclusive_minimum is not None and number <= exclusive_minimum:
            raise SchemaValidationError(
                f"{self._context}: field {key!r} must be > {exclusive_minimum}, got {number}"
            )
        if maximum is not None and number > maximum:
            raise SchemaValidationError(
                f"{self._context}: field {key!r} must be <= {maximum}, got {number}"
            )
        return number

    def get_optional_float(
        self,
        key: str,
        *,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        """Read a float that may be absent or ``null``, falling back to ``default``.

        Used only where the default is genuinely meaningful -- a local configuration
        transmitting ``0.0`` megabytes, say -- never to paper over a missing required field.
        """
        if self._data.get(key) is None:
            return default
        return self.get_float(key, minimum=minimum, maximum=maximum)

    def get_bool(self, key: str) -> bool:
        value = self._require(key)
        if not isinstance(value, bool):
            raise SchemaValidationError(
                f"{self._context}: field {key!r} must be a boolean, got {type(value).__name__}"
            )
        return value

    def get_fraction(self, key: str) -> float:
        """Read a float constrained to [0, 1]."""
        return self.get_float(key, minimum=0.0, maximum=1.0)

    def get_passthrough(self, key: str) -> Any:
        """Return a value unvalidated, for payloads whose shape this layer cannot know.

        The one legitimate use is a task-specific prediction payload: an instance mask set
        for human search, boxes for detection. The schema layer would have to grow a branch
        per task to check them, which is exactly the coupling the architecture forbids, so
        the task's evidence tracker validates its own payloads instead.

        Not an escape hatch. Every field whose shape *is* known to this layer is validated.
        """
        return self._data.get(key)

    def get_enum(self, key: str, enum_type: type[_EnumT]) -> _EnumT:
        value = self.get_str(key)
        try:
            return enum_type(value)
        except ValueError:
            permitted = sorted(member.value for member in enum_type)
            raise SchemaValidationError(
                f"{self._context}: field {key!r} must be one of {permitted}, got {value!r}"
            ) from None

    def get_str_tuple(
        self,
        key: str,
        *,
        allow_empty: bool = False,
        unique: bool = True,
    ) -> tuple[str, ...]:
        value = self._require(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise SchemaValidationError(f"{self._context}: field {key!r} must be a list of strings")
        items: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise SchemaValidationError(
                    f"{self._context}: field {key!r}[{index}] must be a non-empty string"
                )
            items.append(item)
        if not allow_empty and not items:
            raise SchemaValidationError(f"{self._context}: field {key!r} must not be empty")
        if unique and len(set(items)) != len(items):
            duplicates = sorted({item for item in items if items.count(item) > 1})
            raise SchemaValidationError(
                f"{self._context}: field {key!r} contains duplicate entries {duplicates}"
            )
        return tuple(items)

    def get_object(self, key: str, *, allowed_fields: Iterable[str]) -> DocumentReader:
        value = self._require(key)
        if not isinstance(value, Mapping):
            raise SchemaValidationError(f"{self._context}: field {key!r} must be an object")
        return DocumentReader(
            value, context=f"{self._context}.{key}", allowed_fields=allowed_fields
        )

    def get_id_keyed_object(
        self, key: str, *, value_fields: Iterable[str]
    ) -> list[tuple[str, DocumentReader]]:
        """Read an object whose *keys* are identifiers rather than declared field names.

        Used where a document maps IDs to records -- configuration profiles keyed by
        ``config_id``, say. The key set cannot be validated, since any ID is legitimate,
        but each value is validated strictly like any other nested object.

        Returns ``(identifier, reader)`` pairs whose context names the key, so an error
        inside one record still says which record.
        """
        value = self._require(key)
        if not isinstance(value, Mapping):
            raise SchemaValidationError(f"{self._context}: field {key!r} must be an object")
        entries: list[tuple[str, DocumentReader]] = []
        for identifier, record in value.items():
            if not isinstance(identifier, str) or not identifier.strip():
                raise SchemaValidationError(
                    f"{self._context}: field {key!r} keys must be non-empty strings"
                )
            if not isinstance(record, Mapping):
                raise SchemaValidationError(
                    f"{self._context}: field {key!r}[{identifier!r}] must be an object"
                )
            entries.append(
                (
                    identifier,
                    DocumentReader(
                        record,
                        context=f"{self._context}.{key}[{identifier!r}]",
                        allowed_fields=value_fields,
                    ),
                )
            )
        return entries

    def get_object_list(self, key: str, *, allowed_fields: Iterable[str]) -> list[DocumentReader]:
        value = self._require(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise SchemaValidationError(f"{self._context}: field {key!r} must be a list of objects")
        if not value:
            raise SchemaValidationError(f"{self._context}: field {key!r} must not be empty")
        readers: list[DocumentReader] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise SchemaValidationError(
                    f"{self._context}: field {key!r}[{index}] must be an object"
                )
            readers.append(
                DocumentReader(
                    item,
                    context=f"{self._context}.{key}[{index}]",
                    allowed_fields=allowed_fields,
                )
            )
        return readers

    def get_scalar_mapping(self, key: str) -> dict[str, str | int | float | bool | None]:
        """Read an open-ended ``parameters``-style map of JSON scalars.

        Absent or ``null`` yields an empty map. Values are restricted to scalars so that
        a strategy parameter stays inspectable metadata rather than arbitrary structure.
        """
        value = self._data.get(key)
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise SchemaValidationError(f"{self._context}: field {key!r} must be an object")
        result: dict[str, str | int | float | bool | None] = {}
        for name, item in value.items():
            if not isinstance(name, str):
                raise SchemaValidationError(f"{self._context}: field {key!r} keys must be strings")
            if not isinstance(item, _SCALAR_TYPES):
                raise SchemaValidationError(
                    f"{self._context}: field {key!r}[{name!r}] must be a JSON scalar, "
                    f"got {type(item).__name__}"
                )
            result[name] = item
        return result


def read_json_object(path: Path) -> Mapping[str, Any]:
    """Parse ``path`` and confirm it contains a JSON object."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SchemaValidationError(f"{path}: cannot be read ({error})") from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SchemaValidationError(f"{path}: invalid JSON ({error})") from error
    if not isinstance(payload, Mapping):
        raise SchemaValidationError(
            f"{path}: top level must be a JSON object, got {type(payload).__name__}"
        )
    return payload


def _migrate(payload: Mapping[str, Any], *, declared_version: str) -> Mapping[str, Any]:
    """Bring a parsed document up to :data:`~missionbench.SCHEMA_VERSION`.

    V1 supports exactly one version, so this is the identity function. It exists as the
    named seam for future migrations: when ``SUPPORTED_SCHEMA_VERSIONS`` grows, the
    version-specific upgrade steps belong here and every loader inherits them at once.
    """
    del declared_version
    return payload


def open_document(
    path: Path,
    *,
    document_type: str,
    allowed_fields: Iterable[str],
) -> DocumentReader:
    """Load, version-check, and wrap a specification file for validated field access.

    Args:
        path: File to read.
        document_type: Schema name used in error messages, e.g. ``"Contract"``.
        allowed_fields: Every permitted top-level field, excluding ``schema_version``.

    Raises:
        SchemaVersionError: The declared version is unsupported or missing.
        SchemaValidationError: The file is unreadable, is not a JSON object, or carries
            an unknown top-level field.
    """
    payload = read_json_object(path)
    context = f"{path.name} -> {document_type}"

    declared = payload.get(_SCHEMA_VERSION_KEY)
    if declared is None:
        raise SchemaVersionError(
            f"{context}: missing required field {_SCHEMA_VERSION_KEY!r}. "
            f"Every specification file must declare its schema version."
        )
    if not isinstance(declared, str):
        raise SchemaVersionError(
            f"{context}: {_SCHEMA_VERSION_KEY!r} must be a string, got {type(declared).__name__}"
        )
    if declared not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaVersionError(
            f"{context}: unsupported {_SCHEMA_VERSION_KEY} {declared!r}; "
            f"this release supports {sorted(SUPPORTED_SCHEMA_VERSIONS)}. "
            "Old files are never silently reinterpreted."
        )

    payload = _migrate(payload, declared_version=declared)
    return DocumentReader(
        payload,
        context=context,
        allowed_fields=(*allowed_fields, _SCHEMA_VERSION_KEY),
    )
