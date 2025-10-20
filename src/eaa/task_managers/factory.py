import importlib
import inspect
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Iterable, List, Type

from eaa.task_managers.base import BaseTaskManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskManagerMethodSpec:
    """Metadata describing a callable entrypoint on a task manager."""

    name: str
    signature: str
    doc: str


@dataclass(frozen=True)
class TaskManagerSpec:
    """Metadata describing a task manager class."""

    name: str
    qualified_name: str
    module: str
    doc: str
    init_signature: str
    cls: Type[BaseTaskManager]
    methods: List[TaskManagerMethodSpec]


def get_task_manager_specs(refresh: bool = False) -> Dict[str, TaskManagerSpec]:
    """Return discovered task managers keyed by class name."""
    if refresh:
        _cached_specs.cache_clear()
    return dict(_cached_specs())


def get_task_manager_spec(name: str) -> TaskManagerSpec:
    """Return metadata for the requested task manager."""
    specs = get_task_manager_specs()
    try:
        return specs[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown task manager '{name}'. Available managers: {', '.join(sorted(specs))}"
        ) from exc


def get_task_manager_class(name: str) -> Type[BaseTaskManager]:
    """Return the task manager class for the given name."""
    return get_task_manager_spec(name).cls


@lru_cache(maxsize=1)
def _cached_specs() -> MappingProxyType:
    return MappingProxyType(_discover_task_managers())


def _discover_task_managers() -> Dict[str, TaskManagerSpec]:
    specs: Dict[str, TaskManagerSpec] = {}
    for module_name in _iter_task_manager_module_names():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - discovery must be resilient
            logger.debug("Skipping task manager module '%s': %s", module_name, exc)
            continue
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if not _is_task_manager_candidate(cls):
                continue
            methods = _collect_class_methods(cls)
            spec = TaskManagerSpec(
                name=name,
                qualified_name=f"{cls.__module__}.{cls.__qualname__}",
                module=cls.__module__,
                doc=inspect.getdoc(cls) or "",
                init_signature=_format_init_signature(cls),
                cls=cls,
                methods=methods,
            )
            specs[name] = spec
    return specs


def _task_manager_package_prefix() -> str:
    return __name__.rsplit(".", 1)[0]


def _task_manager_root() -> Path:
    return Path(__file__).resolve().parent


def _iter_task_manager_module_names() -> Iterable[str]:
    root = _task_manager_root()
    prefix = _task_manager_package_prefix()
    for path in root.rglob("*.py"):
        if path.name.startswith("_"):
            continue
        module_path = path.relative_to(root).with_suffix("")
        module_name = ".".join(module_path.parts)
        fq_module_name = f"{prefix}.{module_name}"
        if fq_module_name == __name__:
            continue
        yield fq_module_name


def _is_task_manager_candidate(cls: Type[BaseTaskManager]) -> bool:
    if not issubclass(cls, BaseTaskManager):
        return False
    if cls is BaseTaskManager:
        return False
    if cls.__module__ == "eaa.task_managers.base":
        return False
    if cls.__name__.endswith("BaseTaskManager"):
        return False
    return True


def _collect_class_methods(cls: Type[BaseTaskManager]) -> List[TaskManagerMethodSpec]:
    """Collect all methods that start with 'run' from the class.
    """
    methods: List[TaskManagerMethodSpec] = []
    for attr_name, attr in cls.__dict__.items():
        if not attr_name.startswith("run"):
            continue
        func = None
        if isinstance(attr, (staticmethod, classmethod)):
            func = attr.__func__  # type: ignore[attr-defined]
        elif inspect.isfunction(attr):
            func = attr
        if func is None:
            continue
        signature = str(inspect.signature(func))
        doc = inspect.getdoc(func) or ""
        methods.append(TaskManagerMethodSpec(name=attr_name, signature=signature, doc=doc))
    methods.sort(key=lambda item: item.name)
    return methods


def _format_init_signature(cls: Type[BaseTaskManager]) -> str:
    try:
        signature = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return "(...)"
    parameters = [
        str(param)
        for param in signature.parameters.values()
        if param.name != "self"
    ]
    return f"({', '.join(parameters)})"
