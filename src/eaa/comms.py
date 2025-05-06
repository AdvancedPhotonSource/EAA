import os
import logging


logger = logging.getLogger(__name__)


def get_api_key(model_name: str, model_base_url: str) -> str:
    if is_openai_model(model_name, model_base_url):
        return get_openai_api_key()
    elif is_openrouter_model(model_name, model_base_url):
        return get_openrouter_api_key()
    else:
        logger.info(
            f"Could not identify the provider of model {model_name}; returning NotRequired for API key."
        )
        return "NotRequired"


def is_openai_model(model_name: str, model_base_url: str) -> bool:
    return ("gpt" in model_name and model_base_url is None) or ("api.openai.com" in model_base_url)


def is_openrouter_model(model_name: str, model_base_url: str) -> bool:
    return "openrouter.ai" in model_base_url


def get_openai_api_key():
    if os.environ.get('OPENAI_API_KEY'):
        return os.environ['OPENAI_API_KEY']
    else:
        raise ValueError('OPENAI_API_KEY is not set')
    
    
def get_openrouter_api_key():
    if os.environ.get('OPENROUTER_API_KEY'):
        return os.environ['OPENROUTER_API_KEY']
    else:
        raise ValueError('OPENROUTER_API_KEY is not set')
