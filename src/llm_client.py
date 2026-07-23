"""
LLM client abstraction layer. Supports DeepSeek API and local models.

Author: AI Scientist
Date: 2026-05-27
"""

import requests
import logging
import time
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting API and local backends."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        temperature: float = 0.05,
        max_tokens: int = 2048,
        max_retries: int = 3,
        retry_delay: float = 5.0
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.available = bool(self.api_key)

        if self.available:
            logger.info(f"LLMClient initialized with {model} at {api_base}")
        else:
            logger.warning("LLMClient: no API key configured, will fall back to mock mode")

    def _call_api(self, messages: list, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Optional[str]:
        if not self.available:
            return None

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                elif resp.status_code == 429:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited (429), waiting {wait:.0f}s (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(wait)
                else:
                    logger.error(f"API error {resp.status_code}: {resp.text[:500]}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout (attempt {attempt+1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"API call failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return None

    def complete(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Optional[str]:
        """Send a prompt and get completion."""
        messages = [{"role": "user", "content": prompt}]
        return self._call_api(messages, temperature, max_tokens)

    def chat(self, messages: list, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Optional[str]:
        """Send chat messages and get response."""
        return self._call_api(messages, temperature, max_tokens)
