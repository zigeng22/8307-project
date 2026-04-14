"""
OpenRouter API wrapper — unified gateway for GPT-4o, Claude, and other models.
"""
import time
from typing import List, Dict

from models.base import BaseModel


# OpenRouter model ID mapping
OPENROUTER_MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-2024-11-20": "openai/gpt-4o-2024-11-20",
    "claude-3-5-sonnet-20241022": "anthropic/claude-sonnet-4",
    "claude-3.5-sonnet": "anthropic/claude-sonnet-4",
}


class OpenRouterModel(BaseModel):
    """OpenRouter API wrapper — one key for all models."""

    def __init__(self, model_name: str, api_key: str = ""):
        super().__init__(model_name)
        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        # resolve model ID
        self.model_id = OPENROUTER_MODELS.get(model_name, model_name)

    def generate(self, messages: List[Dict[str, str]],
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise e
