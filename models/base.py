"""
Abstract base class for all model wrappers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict


class BaseModel(ABC):
    """Unified interface that all model wrappers must implement."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]],
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        """Generate a response given chat messages.
        
        Args:
            messages: list of {"role": ..., "content": ...}
            max_tokens: max output tokens
            temperature: sampling temperature (0 = greedy)
        
        Returns:
            model response string
        """
        ...

    def batch_generate(self, messages_list: List[List[Dict[str, str]]],
                       max_tokens: int = 1024,
                       temperature: float = 0.0) -> List[str]:
        """Default: sequential generation. Subclasses can override for batch."""
        results = []
        for msgs in messages_list:
            results.append(self.generate(msgs, max_tokens, temperature))
        return results
