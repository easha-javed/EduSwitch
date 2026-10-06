"""Tiny LLM client over plain HTTP (Groq or Gemini). Keys come from .env. Check current model names in the provider docs."""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()


def chat(prompt: str, provider: str = "groq", system: str = "", temperature: float = 0.8, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            if provider == "groq":
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
                    json={"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), "temperature": temperature,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]},
                    timeout=60)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": os.environ["GEMINI_API_KEY"]},
                json={"systemInstruction": {"parts": [{"text": system}]},
                      "contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": temperature}},
                timeout=60)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:  # rate limits and flaky network
            print("LLM call failed:", e)
            time.sleep(2 ** attempt * 3)
    raise RuntimeError("LLM call failed after retries")
