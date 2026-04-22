"""
llm_service.py - Provider-agnostic LLM service adapter over HTTP APIs.

This module keeps LLM usage optional and non-blocking for FastAPI by
preserving the sync generate API and async to_thread wrapper used by the
rest of the engine.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from backend.config import (
    LLM_API_BASE_URL,
    LLM_API_KEY,
    LLM_CHAT_ENDPOINT,
    LLM_DEFAULT_TEMPERATURE,
    LLM_ENABLED,
    LLM_HEALTH_ENDPOINT,
    LLM_MAX_CALLS_PER_MINUTE,
    LLM_MODEL_NAME,
    LLM_MIN_INTERVAL_MS,
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    logger,
)
from backend.llm.guardrails import strip_think_blocks

class LLMService:
    """Thin HTTP adapter for chat-completion style providers.

    Supported formats:
    - ``ollama`` (`/api/chat`)
    - ``llamacpp_server`` / ``openai_compatible`` (`/v1/chat/completions`)
    """

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(
        self,
        provider: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize provider settings and perform a connectivity check.

        Args:
            provider: LLM provider name.
            base_url: Provider base URL.
            model_name: Selected model identifier on provider side.
        """
        self._provider: str = (provider or LLM_PROVIDER).strip().lower()
        self._base_url: str = (base_url or LLM_API_BASE_URL).rstrip("/")
        self._model_name: str = model_name or LLM_MODEL_NAME
        self._api_key: str = LLM_API_KEY
        self._available: bool = False
        self._enabled: bool = LLM_ENABLED

        if self._provider == "ollama":
            self._health_endpoint = LLM_HEALTH_ENDPOINT or "/api/tags"
            self._chat_endpoint = LLM_CHAT_ENDPOINT or "/api/chat"
        else:
            # Default to OpenAI-compatible endpoints.
            self._health_endpoint = LLM_HEALTH_ENDPOINT or "/v1/models"
            self._chat_endpoint = LLM_CHAT_ENDPOINT or "/v1/chat/completions"

        # Rate-limiting state
        self._last_call_time: float = 0.0
        self._calls_this_minute: int = 0
        self._minute_start: float = time.monotonic()

        self._refresh_availability()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Whether provider connectivity checks pass and LLM is enabled."""
        return self._enabled and self._available

    # ── Public API ────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        temperature: float = LLM_DEFAULT_TEMPERATURE,
        max_tokens: int = 1500,
        stop: list[str] | None = None,
    ) -> str | None:
        """Run synchronous text generation.

        Args:
            prompt: The full prompt string.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stop: Optional stop sequences.

        Returns:
            Generated text, or ``None`` on failure / unavailability.
        """
        if not self._enabled:
            return None

        if not self.available:
            logger.warning("LLM generate called but provider is unavailable")
            return None

        if not self._check_rate_limit():
            logger.warning("LLM rate limit exceeded — skipping call")
            return None

        try:
            self._update_rate_tracking()

            messages = self._build_messages(prompt)
            text = self._request_generation(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or [],
            )
            if text is None:
                return None
            text = text.strip()

            text = strip_think_blocks(text)

            logger.info(
                "LLM generated %d chars (temp=%.2f, max_tok=%d)",
                len(text),
                temperature,
                max_tokens,
            )
            return text if text else None
        except Exception as exc:
            logger.error("LLM generation error: %s", exc)
            return None

    async def async_generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str | None:
        """Async wrapper — delegates to :meth:`generate` via ``asyncio.to_thread``.

        Accepts the same keyword arguments as :meth:`generate`.
        """
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.generate, prompt, **kwargs),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            return result
        except asyncio.TimeoutError:
            logger.warning("LLM async_generate timed out after %ds", LLM_TIMEOUT_SECONDS)
            return None
        except Exception as exc:
            logger.error("LLM async_generate error: %s", exc)
            return None

    def get_status(self) -> dict:
        """Return a diagnostic snapshot of the service state."""
        return {
            "enabled": self._enabled,
            "available": self.available,
            "provider": self._provider,
            "base_url": self._base_url,
            "model_name": self._model_name,
            "calls_this_minute": self._calls_this_minute,
            "last_call_time": self._last_call_time,
            "max_calls": LLM_MAX_CALLS_PER_MINUTE,
            "min_interval_ms": LLM_MIN_INTERVAL_MS,
        }

    def unload(self) -> None:
        """Disable LLM calls for the active runtime instance."""
        self._enabled = False
        self._available = False
        logger.info("LLM service disabled")

    # ── Message formatting ────────────────────────────────────────────────

    @staticmethod
    def _build_messages(prompt: str) -> list[dict[str, str]]:
        """Split a raw prompt into system + user chat messages.

        If the prompt opens with 'You are …' the first paragraph is
        extracted as the system message and the remainder becomes the
        user message.  Otherwise the whole prompt is a single user
        message.
        """
        lines = prompt.strip().split("\n")

        # Detect "You are ..." opening and split at the first blank line
        # or the first markdown heading (##).
        if lines and lines[0].lower().startswith("you are"):
            system_lines: list[str] = []
            user_start = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if i > 0 and (stripped == "" or stripped.startswith("#")):
                    user_start = i
                    break
                system_lines.append(line)
            else:
                # Only one paragraph — treat everything as user
                return [{"role": "user", "content": prompt}]

            system_text = "\n".join(system_lines).strip()
            user_text = "\n".join(lines[user_start:]).strip()
            if system_text and user_text:
                return [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ]

        return [{"role": "user", "content": prompt}]

    # - Provider calls -----------------------------------------------------

    def _request_generation(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stop: list[str],
    ) -> str | None:
        """Send a chat generation request to the configured provider."""
        endpoint = f"{self._base_url}{self._chat_endpoint}"

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        if self._provider == "ollama":
            payload: dict[str, Any] = {
                "model": self._model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if stop:
                payload["options"]["stop"] = stop
        else:
            payload = {
                "model": self._model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
            }

        try:
            with httpx.Client(timeout=LLM_TIMEOUT_SECONDS) as client:
                resp = client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.error("LLM provider request failed: %s", exc)
            self._available = False
            return None

        text = self._extract_text(data)
        if text is None:
            logger.error("LLM provider returned an unsupported payload shape")
            return None

        self._available = True
        return text

    def _extract_text(self, data: Any) -> str | None:
        """Extract assistant text from provider-specific response payloads."""
        if not isinstance(data, dict):
            return None

        # Ollama /api/chat
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content

        # OpenAI-compatible format
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"]
                txt = first.get("text")
                if isinstance(txt, str):
                    return txt

        return None

    # ── Rate limiting ─────────────────────────────────────────────────────

    def _check_rate_limit(self) -> bool:
        """Enforce min interval between calls and max calls/min.

        Returns:
            ``True`` if the call is allowed, ``False`` otherwise.
        """
        now = time.monotonic()

        # Min interval (2 s by default)
        elapsed_ms = (now - self._last_call_time) * 1000
        if self._last_call_time > 0 and elapsed_ms < LLM_MIN_INTERVAL_MS:
            logger.debug(
                "Rate limit: only %d ms since last call (min %d ms)",
                int(elapsed_ms),
                LLM_MIN_INTERVAL_MS,
            )
            return False

        # Rolling per-minute cap
        if now - self._minute_start >= 60.0:
            self._calls_this_minute = 0
            self._minute_start = now

        if self._calls_this_minute >= LLM_MAX_CALLS_PER_MINUTE:
            logger.debug(
                "Rate limit: %d calls this minute (max %d)",
                self._calls_this_minute,
                LLM_MAX_CALLS_PER_MINUTE,
            )
            return False

        return True

    def _update_rate_tracking(self) -> None:
        """Record a successful call for rate-tracking purposes."""
        now = time.monotonic()
        self._last_call_time = now
        if now - self._minute_start >= 60.0:
            self._calls_this_minute = 1
            self._minute_start = now
        else:
            self._calls_this_minute += 1

    def _refresh_availability(self) -> None:
        """Best-effort health check to avoid hard failures during first call."""
        if not self._enabled:
            self._available = False
            return

        endpoint = f"{self._base_url}{self._health_endpoint}"
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            with httpx.Client(timeout=min(LLM_TIMEOUT_SECONDS, 5)) as client:
                resp = client.get(endpoint, headers=headers)
                self._available = resp.status_code < 500
        except Exception as exc:
            self._available = False
            logger.info("LLM provider health check failed at startup: %s", exc)
