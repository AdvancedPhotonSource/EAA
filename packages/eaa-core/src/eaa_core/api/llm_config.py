from dataclasses import dataclass
import warnings

from eaa_core.api.base import BaseConfig


@dataclass
class LLMConfig(BaseConfig):
    """Base class for LLM configurations."""


@dataclass
class OpenAIConfig(LLMConfig):
    """Configuration for OpenAI-compatible chat models."""

    model: str = None
    base_url: str = "https://api.openai.com/v1"
    api_key: str = None


@dataclass
class AskSageConfig(LLMConfig):
    """Configuration for AskSage endpoints."""

    model: str = None
    server_base_url: str = None
    user_base_url: str = None
    api_key: str = None
    email: str = None
    cacert_path: str = None


@dataclass
class ArgoConfig(OpenAIConfig):
    """Configuration for Argo endpoints."""

    base_url: str = "https://apps-dev.inside.anl.gov/argoapi/v1"
    api_key: str = "ARGO_USER"
    user: str | None = None

    def __post_init__(self) -> None:
        """Warn when the deprecated compatibility field is still used."""
        if self.user is not None:
            warnings.warn(
                "`ArgoConfig.user` is deprecated and ignored. "
                "Configure authentication through `api_key` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
