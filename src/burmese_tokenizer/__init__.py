from .tokenizer import BurmeseTokenizer

__version__ = "0.1.0"
__author__ = "janakhpon"
__email__ = "jnovaxer@gmail.com"

__all__ = ["BurmeseTokenizer"]


def load_tokenizer(model_path=None):
    """load tokenizer with default settings"""
    return BurmeseTokenizer(model_path)
