"""CrewAI LLM adapter for structured outputs.

Provides a CrewAI-compatible LLM that produces guaranteed structured
outputs using outlines + MLX.
"""

from typing import Any

from crewai import BaseLLM
from pydantic import BaseModel

from localcrew.integrations.structured_mlx import StructuredMLXGenerator


class StructuredMLXLLM(BaseLLM):
    """CrewAI LLM that produces structured outputs via outlines.

    This LLM adapter wraps the StructuredMLXGenerator to provide
    guaranteed structured outputs when used with CrewAI agents.
    When an output_schema is provided, all responses will be valid
    JSON matching that schema.
    """

    def __init__(
        self,
        model: str | None = None,
        output_schema: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the structured MLX LLM.

        Args:
            model: MLX model ID (uses settings default if not provided)
            output_schema: Pydantic model class for structured output
            temperature: Sampling temperature (currently not used by outlines)
            max_tokens: Maximum tokens to generate
        """
        from localcrew.core.config import settings

        model = model or settings.mlx_model_id
        temperature = temperature if temperature is not None else settings.mlx_temperature

        super().__init__(model=model, temperature=temperature)

        self._output_schema = output_schema
        self._max_tokens = max_tokens or settings.mlx_max_tokens
        self._generator: StructuredMLXGenerator | None = None

    @property
    def generator(self) -> StructuredMLXGenerator:
        """Lazy-load the structured generator."""
        if self._generator is None:
            self._generator = StructuredMLXGenerator(self.model)
        return self._generator

    def call(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Call the model with structured or unstructured output.

        Args:
            messages: Either a string prompt or list of message dicts
            tools: Optional tool definitions (not used)
            callbacks: Optional callbacks (not used)
            available_functions: Optional function mapping (not used)
            **kwargs: Additional arguments from CrewAI (from_task, etc.)

        Returns:
            JSON string if output_schema is set, otherwise raw string response.
            Always returns string for CrewAI compatibility.
        """
        import json

        # Convert messages to prompt string
        prompt = self._format_messages(messages) if isinstance(messages, list) else messages

        if self._output_schema:
            # Generate structured output and serialize to JSON string
            result = self.generator.generate(
                prompt=prompt,
                schema=self._output_schema,
                max_tokens=self._max_tokens,
            )
            return json.dumps(result.model_dump(), indent=2)
        else:
            # Fall back to unstructured text generation
            result = self.generator.generate_text(
                prompt=prompt,
                max_tokens=self._max_tokens,
            )
            return result

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format messages into Qwen2 chat format.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        formatted_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        # Add assistant prefix for generation
        formatted_parts.append("<|im_start|>assistant\n")

        return "\n".join(formatted_parts)

    def supports_function_calling(self) -> bool:
        """Structured output replaces function calling need."""
        return False

    def supports_stop_words(self) -> bool:
        """MLX supports stop sequences."""
        return True

    def get_context_window_size(self) -> int:
        """Qwen2.5 supports 32k context."""
        return 32768
