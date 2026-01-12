"""Knowledge Activation System (KAS) client integration.

Provides search and ingestion capabilities for the personal knowledge base.
All operations are optional and gracefully degrade if KAS is unavailable.
"""

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from localcrew.core.config import settings

logger = logging.getLogger(__name__)


class KASResult(BaseModel):
    """A single search result from KAS."""

    content_id: str = Field(description="Unique content identifier")
    title: str = Field(description="Content title")
    content_type: str = Field(description="Type: bookmark|youtube|file|note|research")
    score: float = Field(ge=0.0, le=1.0, description="Relevance score")
    chunk_text: str | None = Field(default=None, description="Matched text chunk")
    source_ref: str | None = Field(default=None, description="Original source reference")


class KASClient:
    """Client for Knowledge Activation System integration.

    Provides both async and sync methods for flexibility:
    - Async methods (search, ingest_research) for use in async contexts
    - Sync methods (search_sync) for use in sync contexts like CrewAI flows

    All operations are designed to fail gracefully - errors are logged
    but don't raise exceptions.
    """

    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        """Initialize KAS client.

        Args:
            base_url: KAS API base URL. Defaults to settings.kas_base_url.
            timeout: Request timeout in seconds. Defaults to settings.kas_timeout.
        """
        self.base_url = (base_url or settings.kas_base_url).rstrip("/")
        self.timeout = timeout or settings.kas_timeout
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None

    def _get_headers(self) -> dict[str, str]:
        """Get common headers for requests."""
        headers = {"Content-Type": "application/json"}
        if settings.kas_api_key:
            headers["Authorization"] = f"Bearer {settings.kas_api_key}"
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
        return self._client

    def _get_sync_client(self) -> httpx.Client:
        """Get or create the sync httpx client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
        return self._sync_client

    async def close(self) -> None:
        """Close the HTTP clients."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    def close_sync(self) -> None:
        """Close the sync HTTP client."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    async def search(self, query: str, limit: int = 10) -> list[KASResult]:
        """Search KAS for relevant knowledge.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of KASResult objects, empty list on error
        """
        try:
            client = await self._get_client()
            response = await client.get(
                "/api/v1/search",
                params={"q": query, "limit": limit},
            )
            response.raise_for_status()

            data = response.json()
            results = []
            for item in data.get("results", data if isinstance(data, list) else []):
                try:
                    results.append(KASResult(**item))
                except Exception as e:
                    logger.warning(f"Failed to parse KAS result: {e}")
            return results

        except httpx.HTTPStatusError as e:
            logger.warning(f"KAS search failed with status {e.response.status_code}: {e}")
            return []
        except httpx.RequestError as e:
            logger.warning(f"KAS search request failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"Unexpected KAS search error: {e}")
            return []

    def search_sync(self, query: str, limit: int = 10) -> list[KASResult]:
        """Search KAS for relevant knowledge (synchronous version).

        Use this method in sync contexts like CrewAI flow methods
        where async/await is not available.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of KASResult objects, empty list on error
        """
        try:
            client = self._get_sync_client()
            response = client.get(
                "/api/v1/search",
                params={"q": query, "limit": limit},
            )
            response.raise_for_status()

            data = response.json()
            results = []
            for item in data.get("results", data if isinstance(data, list) else []):
                try:
                    results.append(KASResult(**item))
                except Exception as e:
                    logger.warning(f"Failed to parse KAS result: {e}")
            return results

        except httpx.HTTPStatusError as e:
            logger.warning(f"KAS search failed with status {e.response.status_code}: {e}")
            return []
        except httpx.RequestError as e:
            logger.warning(f"KAS search request failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"Unexpected KAS search error: {e}")
            return []

    async def ingest_research(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Ingest a research report into KAS.

        Args:
            title: Report title
            content: Full report content (markdown)
            tags: Optional list of tags
            metadata: Optional metadata dictionary

        Returns:
            Content ID if successful, None on error
        """
        try:
            client = await self._get_client()
            response = await client.post(
                "/api/v1/ingest/research",
                json={
                    "title": title,
                    "content": content,
                    "tags": tags or [],
                    "metadata": metadata or {},
                },
            )
            response.raise_for_status()

            data = response.json()
            content_id = data.get("content_id") or data.get("id")
            if content_id:
                logger.info(f"Research ingested to KAS: {content_id}")
            return content_id

        except httpx.HTTPStatusError as e:
            logger.warning(f"KAS ingest failed with status {e.response.status_code}: {e}")
            return None
        except httpx.RequestError as e:
            logger.warning(f"KAS ingest request failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected KAS ingest error: {e}")
            return None

    async def health_check(self) -> bool:
        """Check if KAS is reachable.

        Returns:
            True if KAS responds to health check, False otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/v1/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"KAS health check failed: {e}")
            return False


# Singleton instance
_kas_instance: KASClient | None = None


def get_kas() -> KASClient | None:
    """Get KAS client if enabled, None otherwise.

    Returns singleton instance for efficient connection reuse.

    Returns:
        KASClient instance if kas_enabled=True, None otherwise
    """
    if not settings.kas_enabled:
        return None

    global _kas_instance
    if _kas_instance is None:
        _kas_instance = KASClient()
    return _kas_instance


async def reset_kas() -> None:
    """Reset the KAS singleton (for testing)."""
    global _kas_instance
    if _kas_instance:
        await _kas_instance.close()
        _kas_instance = None
