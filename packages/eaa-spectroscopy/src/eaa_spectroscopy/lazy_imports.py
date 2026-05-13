"""Helpers for module-level lazy imports."""

from importlib import import_module


def resolve_lazy_export(
    exports: dict[str, str],
    namespace: dict[str, object],
    module_name: str,
    name: str,
) -> object:
    """Resolve and cache a lazily exported symbol.

    Parameters
    ----------
    exports : dict[str, str]
        Mapping from exported symbol names to importable module paths.
    namespace : dict[str, object]
        Module globals where the resolved symbol should be cached.
    module_name : str
        Name of the module exposing the lazy export.
    name : str
        Requested attribute name.

    Returns
    -------
    object
        Resolved exported object.
    """
    if name not in exports:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    module = import_module(exports[name])
    value = getattr(module, name)
    namespace[name] = value
    return value


def list_lazy_exports(namespace: dict[str, object], exports: list[str]) -> list[str]:
    """Return module attributes including lazy exports.

    Parameters
    ----------
    namespace : dict[str, object]
        Module globals.
    exports : list[str]
        Public lazy export names.

    Returns
    -------
    list[str]
        Sorted module attribute names.
    """
    return sorted([*namespace, *exports])
