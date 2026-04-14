"""
Test different OpenRouter models to isolate provider vs account issue.
Usage: set OPENROUTER_API_KEY env var first, then run this script.
"""
import os
from openai import OpenAI

key = os.environ.get("OPENROUTER_API_KEY", "")
if not key:
    key = input("Paste your OpenRouter API key: ").strip()

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

models = [
    # strong paid models (potential replacements for GPT-4o / Claude)
    ("google/gemini-2.0-flash-001", "Google Gemini 2.0 Flash"),
    ("google/gemini-2.5-pro-preview", "Google Gemini 2.5 Pro"),
    ("deepseek/deepseek-chat-v3-0324", "DeepSeek V3"),
    ("deepseek/deepseek-r1", "DeepSeek R1"),
    ("mistralai/mistral-large-2411", "Mistral Large"),
    ("cohere/command-r-plus-08-2024", "Cohere Command R+"),
    ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B"),
]

for model_id, provider in models:
    try:
        r = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=5,
        )
        print(f"  OK  {provider:20s} {model_id} -> {r.choices[0].message.content}")
    except Exception as e:
        err = str(e)
        short = err[:120] if len(err) > 120 else err
        print(f"  FAIL {provider:20s} {model_id} -> {short}")
