"""
Wrappers for closed-source API models (OpenAI, Anthropic).
"""
import time
from typing import List, Dict

from models.base import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI API wrapper (GPT-4o etc.)."""

    def __init__(self, model_name: str = "gpt-4o", api_key: str = ""):
        super().__init__(model_name)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_name

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


class AnthropicModel(BaseModel):
    """Anthropic API wrapper (Claude 3.5 Sonnet etc.)."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022",
                 api_key: str = ""):
        super().__init__(model_name)
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_id = model_name

    def generate(self, messages: List[Dict[str, str]],
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        # Anthropic uses system param separately
        system_msg = ""
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                chat_msgs.append(m)

        for attempt in range(3):
            try:
                kwargs = {
                    "model": self.model_id,
                    "max_tokens": max_tokens,
                    "messages": chat_msgs,
                }
                if system_msg:
                    kwargs["system"] = system_msg
                if temperature > 0:
                    kwargs["temperature"] = temperature
                resp = self.client.messages.create(**kwargs)
                return resp.content[0].text.strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise e
