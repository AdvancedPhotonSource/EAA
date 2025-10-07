from dataclasses import dataclass

from eaa.api.base import BaseConfig


@dataclass
class LLMConfig(BaseConfig):
    """A base class for LLM configurations.
    
    This class is used to store the configuration for an LLM.
    """
    pass


@dataclass
class OpenAIConfig(LLMConfig):
    model: str = None
    """The name of the model to use."""
    
    base_url: str = "https://api.openai.com/v1"
    """The base URL of the inference endpoint."""
    
    api_key: str = None
    """The API key."""


@dataclass
class AskSageConfig(LLMConfig):
    model: str = None
    """The name of the model to use."""
    
    server_base_url: str = None
    """The base URL for LLM queries."""
    
    user_base_url: str = None
    """The base URL for user-related queries."""
    
    api_key: str = None
    """The API key."""
    
    email: str = None
    """The email of the user."""
    
    cacert_path: str = None
    """The path to the CA certificate file (*.pem)."""
