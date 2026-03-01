"""
llm_service.py — LLM service wrapper for local GGUF model inference.

Wraps llama-cpp-python with rate limiting, async support, and graceful
degradation when the model is unavailable.  The entire game works
without LLM; this module simply makes it *optional*.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from backend.config import (
    LLM_CONTEXT_SIZE,
    LLM_DEFAULT_TEMPERATURE,
    LLM_MAX_CALLS_PER_MINUTE,
    LLM_MAX_PROMPT_TOKENS,
    LLM_MIN_INTERVAL_MS,
    LLM_MODEL_PATH,
    LLM_TIMEOUT_SECONDS,
    logger,
)

# Attempt to import llama-cpp-python; if missing, the service still
# instantiates but will report available=False.
try:
    from llama_cpp import Llama  # type: ignore[import-untyped]

    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    Llama = None  # type: ignore[assignment,misc]


class LLMService:
    """Thin wrapper around a local GGUF model loaded via llama-cpp-python.

    Key behaviours:
    * If the model file is missing or the library is absent ➜ ``available == False``.
    * Rate-limited: min 2 s between calls, max 20 calls/min.
    * ``async_generate`` delegates to ``asyncio.to_thread`` so the event
      loop is never blocked.
    """

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(
        self,
        model_path: str | None = None,
        context_size: int = LLM_CONTEXT_SIZE,
    ) -> None:
        """Load the GGUF model if possible.

        Args:
            model_path: Filesystem path to the ``.gguf`` file.
                Falls back to ``LLM_MODEL_PATH`` from config.
            context_size: Context window size for the model.
        """
        self._model_path: str = model_path or LLM_MODEL_PATH
        self._context_size: int = context_size
        self._model: Any = None
        self._available: bool = False

        # Rate-limiting state
        self._last_call_time: float = 0.0
        self._calls_this_minute: int = 0
        self._minute_start: float = time.monotonic()

        self._load_model()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        return self._available

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
        if not self._available:
            logger.warning("LLM generate called but model is not available")
            return None

        if not self._check_rate_limit():
            logger.warning("LLM rate limit exceeded — skipping call")
            return None

        try:
            self._update_rate_tracking()
            result = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or [],
                echo=False,
            )
            text: str = result["choices"][0]["text"].strip()
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
            "available": self._available,
            "model_path": self._model_path,
            "context_size": self._context_size,
            "calls_this_minute": self._calls_this_minute,
            "last_call_time": self._last_call_time,
            "llama_cpp_installed": _LLAMA_AVAILABLE,
        }

    def unload(self) -> None:
        """Free model resources and mark the service as unavailable."""
        if self._model is not None:
            try:
                del self._model
            except Exception:
                pass
            self._model = None
        self._available = False
        logger.info("LLM model unloaded")

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

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Attempt to load the GGUF model file."""
        if not _LLAMA_AVAILABLE:
            logger.warning(
                "llama-cpp-python is not installed — LLM service disabled"
            )
            self._available = False
            return

        import pathlib

        path = pathlib.Path(self._model_path)
        if not path.exists():
            logger.warning(
                "Model file not found at %s — LLM service disabled",
                self._model_path,
            )
            self._available = False
            return

        try:
            self._model = Llama(
                model_path=str(path),
                n_ctx=self._context_size,
                verbose=False,
            )
            self._available = True
            logger.info(
                "LLM model loaded: %s (ctx=%d)", path.name, self._context_size
            )
        except Exception as exc:
            logger.error("Failed to load LLM model: %s", exc)
            self._available = False
