"""LLM backend integration with the shared pipeline initialized first."""
from hlsmith import pipeline as _core
from .client import LLMBackendMixin

__all__ = ["LLMBackendMixin"]
