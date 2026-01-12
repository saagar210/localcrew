"""MLX LLM integration for CrewAI."""

from typing import Any

from crewai import BaseLLM

from localcrew.integrations.mlx import MLXInference


class MLXLLM(BaseLLM):
    """Custom LLM using MLX for local inference on Apple Silicon."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize MLX LLM.

        Args:
            model: Model ID for MLX (e.g., "mlx-community/Qwen2.5-14B-Instruct-4bit")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
        """
        from localcrew.core.config import settings

        # Use settings defaults if not provided
        model = model or settings.mlx_model_id
        temperature = temperature if temperature is not None else settings.mlx_temperature

        super().__init__(model=model, temperature=temperature)

        self._max_tokens = max_tokens or settings.mlx_max_tokens
        self._mlx: MLXInference | None = None

    @property
    def mlx(self) -> MLXInference:
        """Lazy-load MLX inference engine."""
        if self._mlx is None:
            self._mlx = MLXInference()
        return self._mlx

    def call(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | Any:
        """Call the MLX model with the given messages.

        Args:
            messages: Either a string prompt or list of message dicts
            tools: Optional list of tool definitions (not supported by MLX)
            callbacks: Optional callbacks (not used)
            available_functions: Optional function mapping (not used)
            **kwargs: Additional arguments from CrewAI (from_task, etc.)

        Returns:
            Generated text response
        """
        # Convert messages to a single prompt
        if isinstance(messages, str):
            prompt = messages
        else:
            # Format messages into chat prompt
            prompt = self._format_messages(messages)

        # Generate response
        response = self.mlx.generate(
            prompt=prompt,
            max_tokens=self._max_tokens,
            temperature=self.temperature or 0.7,
        )

        return response

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format messages into a chat prompt.

        Uses Qwen2 chat format.
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
        """MLX models don't support function calling natively."""
        return False

    def supports_stop_words(self) -> bool:
        """MLX supports stop sequences."""
        return True

    def get_context_window_size(self) -> int:
        """Return context window size (Qwen2.5 supports 32k)."""
        return 32768
