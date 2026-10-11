"""Version-aware persistence with warnings before Polars expression restoration."""

import io
import pickle
import platform
import warnings
from collections.abc import Callable
from operator import index
from typing import Any, SupportsIndex

import cloudpickle
import numpy as np
import polars as pl
import sklearn

from .exceptions import InconsistentVersionWarning, SerializationError


def _versions() -> dict[str, str]:
    # Import after package initialization, avoiding a cycle through public classes.
    from . import __version__

    return {
        "python": platform.python_version(),
        "polars-learn": __version__,
        "polars": pl.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "cloudpickle": cloudpickle.__version__,
    }


def _restore_expression(
    binary: bytes, json_payload: str | None, original_version: str
) -> pl.Expr:
    try:
        if original_version == pl.__version__:
            return pl.Expr.deserialize(io.BytesIO(binary), format="binary")
        if json_payload is None:
            raise SerializationError("This expression has no JSON representation.")
        return pl.Expr.deserialize(
            io.BytesIO(json_payload.encode("utf-8")), format="json"
        )
    except (Exception, pl.exceptions.PanicException) as exc:
        raise SerializationError(
            f"Cannot restore a Polars expression saved with Polars {original_version} "
            f"using Polars {pl.__version__}: {exc}. Load it in the original "
            "environment or recreate the specification with the current API."
        ) from exc


class _StatePickler(pickle.Pickler):
    def __init__(self, buffer: io.BytesIO, protocol: int, owner: object) -> None:
        super().__init__(buffer, protocol=protocol)
        self.owner = owner

    def persistent_id(self, obj: object) -> str | None:
        return "owner" if obj is self.owner else None

    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, pl.Expr):
            # Keep binary for lossless same-version restores: JSON cannot
            # represent every native literal (notably non-finite scalars).
            binary = obj.meta.serialize(format="binary")
            if not isinstance(binary, bytes):
                raise SerializationError("Polars did not return expression bytes.")
            try:
                json_payload = obj.meta.serialize(format="json")
            except (Exception, pl.exceptions.PanicException):
                json_payload = None
            if json_payload is not None and not isinstance(json_payload, str):
                raise SerializationError("Polars did not return an expression string.")
            # Cross-version loads try JSON; do not silently reinterpret old
            # binary layouts if the current version rejects the JSON plan.
            return _restore_expression, (binary, json_payload, pl.__version__)
        return NotImplemented


class _StateUnpickler(pickle.Unpickler):
    def __init__(self, buffer: io.BytesIO, owner: object) -> None:
        super().__init__(buffer)
        self.owner = owner

    def persistent_load(self, pid: object) -> object:
        if pid != "owner":
            raise pickle.UnpicklingError("Unknown persistent reference in saved state.")
        return self.owner


def _restore_object(
    cls: type[Any], payload: bytes, original_versions: dict[str, str]
) -> Any:
    current_versions = _versions()
    for dependency, original in original_versions.items():
        current = current_versions.get(dependency, "unavailable")
        if original != current:
            warnings.warn(
                InconsistentVersionWarning(cls.__name__, dependency, original, current),
                stacklevel=2,
            )
    # Warnings promoted to errors stop execution before state/Expr deserialization.
    instance: Any = object.__new__(cls)
    try:
        state = _StateUnpickler(io.BytesIO(payload), instance).load()
        if not isinstance(state, dict):
            raise SerializationError("Serialized object state must be a dict.")
        instance.__dict__.update(state)
    except SerializationError:
        raise
    except (Exception, pl.exceptions.PanicException) as exc:
        raise SerializationError(
            f"Cannot restore {cls.__name__}. Saved environment: {original_versions}; "
            f"current environment: {current_versions}. Underlying error: {exc}"
        ) from exc
    return instance


def reduce_object(
    obj: object, protocol: SupportsIndex
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Encode state separately so version checks run before dependent restoration."""
    buffer = io.BytesIO()
    try:
        _StatePickler(buffer, index(protocol), obj).dump(obj.__dict__)
    except (Exception, pl.exceptions.PanicException) as exc:
        raise SerializationError(
            f"Cannot serialize {type(obj).__name__}: {exc}. "
            "Expression callbacks and parameter state must be serializable."
        ) from exc
    return _restore_object, (type(obj), buffer.getvalue(), _versions())
