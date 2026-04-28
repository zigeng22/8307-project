"""
Wrapper for HuggingFace open-source models (Llama, Qwen, Gemma).
Supports both base and LoRA-finetuned checkpoints.
"""
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base import BaseModel


class HFModel(BaseModel):
    """Local HuggingFace model for inference."""

    def __init__(self, model_id: str, lora_path: Optional[str] = None,
                 device: str = "auto", dtype=torch.float16):
        super().__init__(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device,
        )

        # merge LoRA adapter if provided
        if lora_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()

        self.model.eval()

    def generate(self, messages: List[Dict[str, str]],
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        # use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Some templates (e.g., Gemma) don't support a separate system role.
                system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
                non_system = [m for m in messages if m.get("role") != "system"]

                if system_msgs:
                    prefix = "System instructions:\n" + "\n".join(system_msgs)
                    if non_system and non_system[0].get("role") == "user":
                        non_system[0] = {
                            "role": "user",
                            "content": f"{prefix}\n\n{non_system[0].get('content', '')}",
                        }
                    else:
                        non_system.insert(0, {"role": "user", "content": prefix})

                try:
                    input_text = self.tokenizer.apply_chat_template(
                        non_system, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    parts = []
                    for m in messages:
                        parts.append(f"{m['role']}: {m['content']}")
                    parts.append("assistant:")
                    input_text = "\n".join(parts)
        else:
            # fallback: concatenate messages
            parts = []
            for m in messages:
                parts.append(f"{m['role']}: {m['content']}")
            parts.append("assistant:")
            input_text = "\n".join(parts)

        inputs = self.tokenizer(input_text, return_tensors="pt").to(
            self.model.device
        )
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # decode only new tokens
        new_tokens = output_ids[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
