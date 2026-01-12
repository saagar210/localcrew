"""MLX inference integration for local model execution."""

import logging
from typing import Any

from localcrew.core.config import settings

logger = logging.getLogger(__name__)


class MLXInference:
    """MLX-native inference for local LLM execution on Apple Silicon."""

    def __init__(
        self,
        model_id: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self.model_id = model_id or settings.mlx_model_id
        self.max_tokens = max_tokens or settings.mlx_max_tokens
        self.temperature = temperature or settings.mlx_temperature
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy load the MLX model."""
        if self._model is not None:
            return

        try:
            from mlx_lm import load

            logger.info(f"Loading MLX model: {self.model_id}")
            self._model, self._tokenizer = load(self.model_id)
            logger.info("Model loaded successfully")

        except ImportError:
            logger.error("mlx_lm not installed. Install with: pip install mlx-lm")
            raise
        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            # Try fallback model
            if self.model_id != settings.mlx_fallback_model_id:
                logger.info(f"Trying fallback model: {settings.mlx_fallback_model_id}")
                self.model_id = settings.mlx_fallback_model_id
                self._load_model()
            else:
                raise

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """
        Generate text using the MLX model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Sequences that stop generation

        Returns:
            Generated text
        """
        self._load_model()

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        temp = temperature or self.temperature
        sampler = make_sampler(temp=temp)

        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens or self.max_tokens,
            sampler=sampler,
        )

        return response

    def generate_structured(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate structured output using the MLX model.

        Args:
            prompt: Input prompt with JSON formatting instructions
            output_schema: Expected output schema for validation
            max_tokens: Maximum tokens to generate

        Returns:
            Parsed JSON output
        """
        import json

        # Add JSON formatting instructions
        formatted_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{json.dumps(output_schema, indent=2)}

JSON Response:"""

        response = self.generate(
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temp for structured output
        )

        # Extract JSON from response
        try:
            # Find JSON block in response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in response, returning raw")
                return {"raw_response": response}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {"raw_response": response, "parse_error": str(e)}

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        self._load_model()
        tokens = self._tokenizer.encode(text)
        return len(tokens)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


# Singleton instance for reuse
_mlx_instance: MLXInference | None = None


def get_mlx() -> MLXInference:
    """Get or create MLX inference instance."""
    global _mlx_instance
    if _mlx_instance is None:
        _mlx_instance = MLXInference()
    return _mlx_instance
