import os
import logging


logger = logging.getLogger(__name__)


def get_api_key(model_name: str) -> str:
    if is_openai_model(model_name):
        return get_openai_api_key()
    else:
        logger.info(
            f"Could not identify the provider of model {model_name}; returning NotRequired for API key."
        )
        return "NotRequired"


def is_openai_model(model_name: str) -> bool:
    return "gpt" in model_name


def get_openai_api_key():
    if os.environ.get('OPENAI_API_KEY'):
        return os.environ['OPENAI_API_KEY']
    else:
        raise ValueError('OPENAI_API_KEY is not set')
