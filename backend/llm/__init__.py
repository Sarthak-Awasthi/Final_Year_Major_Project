"""
llm — Local LLM integration layer (always optional).

Submodules:
  llm_service   LLM model wrapper with rate limiting
  prompts       Prompt template builders
  guardrails    Output validation and sanitization
  fallback      Template-based fallbacks when LLM is unavailable
"""

from backend.llm.fallback import (
    fallback_checkpoint,
    fallback_dialogue,
    fallback_input_analysis,
    fallback_narration,
)
from backend.llm.guardrails import (
    VALID_EMOTIONS,
    VALID_SOCIAL,
    clamp,
    parse_json_response,
    sanitize_text,
    validate_checkpoint_output,
    validate_dialogue_output,
    validate_input_analysis,
)
from backend.llm.llm_service import LLMService
from backend.llm.prompts import (
    build_checkpoint_prompt,
    build_dialogue_prompt,
    build_input_analysis_prompt,
    build_narration_prompt,
    estimate_tokens,
    truncate_to_budget,
)

__all__ = [
    # Service
    "LLMService",
    # Prompts
    "build_checkpoint_prompt",
    "build_dialogue_prompt",
    "build_input_analysis_prompt",
    "build_narration_prompt",
    "estimate_tokens",
    "truncate_to_budget",
    # Guardrails
    "validate_checkpoint_output",
    "validate_dialogue_output",
    "validate_input_analysis",
    "parse_json_response",
    "sanitize_text",
    "clamp",
    "VALID_EMOTIONS",
    "VALID_SOCIAL",
    # Fallbacks
    "fallback_checkpoint",
    "fallback_dialogue",
    "fallback_input_analysis",
    "fallback_narration",
]
