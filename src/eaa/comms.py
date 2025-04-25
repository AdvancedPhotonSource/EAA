import os


def get_openai_api_key():
    if os.environ.get('OPENAI_API_KEY'):
        return os.environ['OPENAI_API_KEY']
    else:
        raise ValueError('OPENAI_API_KEY is not set')
