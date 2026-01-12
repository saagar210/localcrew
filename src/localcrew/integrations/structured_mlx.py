"""Structured MLX generation using outlines.

Uses the outlines library with native MLX support to generate
structured outputs that are guaranteed to match Pydantic schemas.
"""

from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StructuredMLXGenerator:
    """Generates structured outputs using outlines + MLX.

    This generator uses outlines' native MLX support via `from_mlxlm()`
    to perform constrained token sampling, guaranteeing valid JSON
    output that matches the provided Pydantic schema.
    """

    def __init__(self, model_id: str | None = None) -> None:
        """Initialize the structured generator.

        Args:
            model_id: MLX model ID (e.g., "mlx-community/Qwen2.5-14B-Instruct-4bit")
                     If not provided, uses settings.mlx_model_id
        """
        from localcrew.core.config import settings

        self.model_id = model_id or settings.mlx_model_id
        self._model = None

    def _load_model(self):
        """Lazy-load the outlines MLX model."""
        if self._model is None:
            import mlx_lm
            import outlines

            # mlx_lm.load returns (model, tokenizer) tuple
            # outlines.from_mlxlm expects both as arguments
            self._model = outlines.from_mlxlm(*mlx_lm.load(self.model_id))
        return self._model

    def generate(
        self,
        prompt: str,
        schema: type[T],
        max_tokens: int = 2048,
    ) -> T:
        """Generate structured output matching the schema.

        Args:
            prompt: The prompt to send to the model
            schema: Pydantic model class defining the output structure
            max_tokens: Maximum tokens to generate

        Returns:
            Instance of the schema class with generated values
        """
        import json

        model = self._load_model()
        # Call model with schema as output_type - returns JSON string
        json_str = model(prompt, schema, max_tokens=max_tokens)
        # Parse JSON string and validate with Pydantic
        data = json.loads(json_str)
        return schema(**data)

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        """Generate unstructured text (fallback mode).

        Args:
            prompt: The prompt to send to the model
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text string
        """
        model = self._load_model()
        # Call model without output_type for plain text
        result = model(prompt, max_tokens=max_tokens)
        return result
