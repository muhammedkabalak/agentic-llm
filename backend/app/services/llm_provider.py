"""
LLM Provider Abstraction.

Defines a single `BaseLLMProvider` interface that all concrete providers
(OpenAI, Anthropic, Ollama / local) implement. The rest of the application
talks to providers ONLY through this interface — swapping providers becomes
a configuration change, not a code change.

Design pattern: Strategy + Factory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import LLMProvider, Settings, get_settings
from app.services.logging_service import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------- #
@dataclass
class LLMMessage:
    """A single chat-style message."""
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    """Normalized LLM response across all providers."""
    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: Any = field(default=None, repr=False)


class LLMProviderError(Exception):
    """Raised when an LLM call fails after all retries."""


# --------------------------------------------------------------------- #
# Abstract base
# --------------------------------------------------------------------- #
class BaseLLMProvider(ABC):
    """Abstract LLM provider. Subclasses implement `_generate`."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.timeout = settings.llm_timeout_seconds

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    async def _generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(LLMProviderError),
        reraise=True,
    )
    async def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Public entry-point with retry/backoff."""
        try:
            response = await self._generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            logger.info(
                "llm_call_success",
                provider=self.provider_name,
                model=response.model,
                total_tokens=response.total_tokens,
            )
            return response
        except Exception as exc:
            logger.error(
                "llm_call_failed",
                provider=self.provider_name,
                model=self.model,
                error=str(exc),
            )
            raise LLMProviderError(str(exc)) from exc

    async def stream(
        self,
        messages: list[LLMMessage],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Optional streaming. Default = non-streaming fallback."""
        response = await self.generate(messages, **kwargs)
        yield response.content


# --------------------------------------------------------------------- #
# OpenAI implementation
# --------------------------------------------------------------------- #
class OpenAIProvider(BaseLLMProvider):
    @property
    def provider_name(self) -> str:
        return "openai"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIProvider")

        # Lazy import keeps optional deps optional
        from langchain_openai import ChatOpenAI

        self._client = ChatOpenAI(
            model=self.model,
            api_key=settings.openai_api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

    async def _generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
        )

        role_map = {
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage,
        }
        lc_messages = [role_map[m.role](content=m.content) for m in messages]

        client = self._client
        if temperature is not None or max_tokens is not None:
            client = self._client.bind(
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            )

        result = await client.ainvoke(lc_messages)
        usage = getattr(result, "usage_metadata", {}) or {}

        return LLMResponse(
            content=result.content,
            model=self.model,
            provider=self.provider_name,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            raw=result,
        )


# --------------------------------------------------------------------- #
# Anthropic implementation
# --------------------------------------------------------------------- #
class AnthropicProvider(BaseLLMProvider):
    @property
    def provider_name(self) -> str:
        return "anthropic"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for AnthropicProvider")

        from langchain_anthropic import ChatAnthropic

        self._client = ChatAnthropic(
            model=self.model,
            api_key=settings.anthropic_api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

    async def _generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
        )

        role_map = {
            "system": SystemMessage,
            "user": HumanMessage,
            "assistant": AIMessage,
        }
        lc_messages = [role_map[m.role](content=m.content) for m in messages]

        result = await self._client.ainvoke(lc_messages)
        usage = getattr(result, "usage_metadata", {}) or {}

        return LLMResponse(
            content=result.content,
            model=self.model,
            provider=self.provider_name,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            raw=result,
        )


# --------------------------------------------------------------------- #
# Ollama (local) implementation
# --------------------------------------------------------------------- #
class OllamaProvider(BaseLLMProvider):
    """
    Local LLM via Ollama (https://ollama.com).

    Talks to Ollama's HTTP API directly so we don't need an extra
    `langchain_ollama` dep. Token counts come from Ollama's
    `prompt_eval_count` / `eval_count` fields.
    """

    @property
    def provider_name(self) -> str:
        return "ollama"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._base_url = settings.ollama_base_url.rstrip("/")
        # httpx is already a transitive dep via FastAPI / langchain.
        import httpx

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self.timeout),
        )

    async def _generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "stream": False,
            "options": {
                "temperature": (
                    temperature if temperature is not None else self.temperature
                ),
                "num_predict": (
                    max_tokens if max_tokens is not None else self.max_tokens
                ),
            },
        }

        response = await self._client.post("/api/chat", json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned {response.status_code}: {response.text[:300]}"
            )
        data = response.json()
        # /api/chat returns {"message": {"role": "assistant", "content": "..."},
        #  "prompt_eval_count": ..., "eval_count": ..., ...}
        msg = data.get("message", {})
        content = msg.get("content", "")
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            raw=data,
        )


# --------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------- #
_PROVIDER_REGISTRY: dict[LLMProvider, type[BaseLLMProvider]] = {
    LLMProvider.OPENAI: OpenAIProvider,
    LLMProvider.ANTHROPIC: AnthropicProvider,
    LLMProvider.LOCAL: OllamaProvider,
}


def get_llm_provider(settings: Optional[Settings] = None) -> BaseLLMProvider:
    """Factory that returns the configured LLM provider."""
    settings = settings or get_settings()
    provider_cls = _PROVIDER_REGISTRY.get(settings.llm_provider)
    if provider_cls is None:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
    logger.info(
        "llm_provider_initialized",
        provider=settings.llm_provider.value,
        model=settings.llm_model,
    )
    return provider_cls(settings)
