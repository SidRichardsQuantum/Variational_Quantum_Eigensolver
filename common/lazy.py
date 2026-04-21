from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

LazyExports = dict[str, tuple[str, str]]


def load_export(
    *,
    package_name: str,
    package_globals: dict[str, Any],
    exports: LazyExports,
    name: str,
) -> Any:
    """Resolve and memoize a lazily exported package attribute."""
    try:
        module_name, attr_name = exports[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {package_name!r} has no attribute {name!r}"
        ) from exc

    value = getattr(import_module(module_name), attr_name)
    package_globals[name] = value
    return value


def list_exports(module: ModuleType, exports: LazyExports) -> list[str]:
    """Return names exposed by a package module plus its lazy exports."""
    return sorted(set(vars(module)) | set(exports))
