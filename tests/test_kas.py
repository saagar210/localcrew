"""Tests for KAS client integration."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from localcrew.integrations.kas import (
    KASClient,
    KASResult,
    get_kas,
    reset_kas,
)


class TestKASResult:
    """Tests for KASResult model."""

    def test_kas_result_valid(self):
        """Test KASResult with valid data."""
        result = KASResult(
            content_id="abc123",
            title="Test Document",
            content_type="note",
            score=0.85,
            chunk_text="This is a test chunk of text.",
            source_ref="https://example.com/doc",
        )
        assert result.content_id == "abc123"
        assert result.score == 0.85
        assert result.content_type == "note"

    def test_kas_result_score_bounds(self):
        """Test KASResult enforces score bounds."""
        with pytest.raises(ValueError):
            KASResult(
                content_id="abc123",
                title="Test",
                content_type="note",
                score=1.5,  # Over 1.0
                chunk_text="Test",
            )

        with pytest.raises(ValueError):
            KASResult(
                content_id="abc123",
                title="Test",
                content_type="note",
                score=-0.1,  # Negative
                chunk_text="Test",
            )

    def test_kas_result_defaults(self):
        """Test KASResult default values."""
        result = KASResult(
            content_id="abc123",
            title="Test",
            content_type="bookmark",
            score=0.5,
            chunk_text="Test content",
        )
        assert result.source_ref is None
        assert result.chunk_text == "Test content"


class TestKASClient:
    """Tests for KASClient."""

    @pytest.fixture
    def kas_client(self):
        """Create a KAS client for testing."""
        return KASClient(base_url="http://localhost:8000", timeout=5.0)

    @pytest.mark.asyncio
    async def test_search_success(self, kas_client):
        """Test successful search request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "content_id": "doc1",
                    "title": "Test Document 1",
                    "content_type": "note",
                    "score": 0.9,
                    "chunk_text": "Relevant text about the query",
                    "source_ref": "",
                },
                {
                    "content_id": "doc2",
                    "title": "Test Document 2",
                    "content_type": "bookmark",
                    "score": 0.75,
                    "chunk_text": "Another relevant text",
                    "source_ref": "https://example.com",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(kas_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            results = await kas_client.search("test query", limit=5)

            assert len(results) == 2
            assert results[0].content_id == "doc1"
            assert results[0].score == 0.9
            assert results[1].content_type == "bookmark"

            mock_client.get.assert_called_once_with(
                "/api/v1/search",
                params={"q": "test query", "limit": 5},
            )

    @pytest.mark.asyncio
    async def test_search_empty_results(self, kas_client):
        """Test search with no results."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(kas_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            results = await kas_client.search("obscure query")

            assert results == []

    @pytest.mark.asyncio
    async def test_search_http_error(self, kas_client):
        """Test search handles HTTP errors gracefully."""
        with patch.object(kas_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            error = httpx.HTTPStatusError(
                "Server Error", request=MagicMock(), response=mock_response
            )
            mock_client.get = AsyncMock(side_effect=error)
            mock_get_client.return_value = mock_client

            results = await kas_client.search("test query")

            assert results == []

    @pytest.mark.asyncio
    async def test_search_connection_error(self, kas_client):
        """Test search handles connection errors gracefully."""
        with patch.object(kas_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.RequestError("Connection refused")
            )
            mock_get_client.return_value = mock_client

            results = await kas_client.search("test query")

            assert results == []

    @pytest.mark.asyncio
    async def test_ingest_research_success(self, kas_client):
        """Test successful research ingestion."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"content_id": "new_doc_123"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(kas_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            content_id = await kas_client.ingest_research(
                title="Test Research",
                content="# Research Report\n\nThis is test content.",
                tags=["research", "test"],
                metadata={"confidence": 85},
            )

            assert content_id == "new_doc_123"
            mock_client.post.assert_called_once_with(
                "/api/v1/ingest/research",
                json={
                    "title": "Test Research",
                    "content": "# Research Report\n\nThis is test content.",
                    "tags": ["research", "test"],
                    "metadata": {"confidence": 85},
                },
            )

    @pytest.mark.asyncio
    async def test_ingest_research_failure(self, kas_client):
        """Test ingestion handles errors gracefully."""
        with patch.object(kas_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 400
            error = httpx.HTTPStatusError(
                "Bad Request", request=MagicMock(), response=mock_response
            )
            mock_client.post = AsyncMock(side_effect=error)
            mock_get_client.return_value = mock_client

            content_id = await kas_client.ingest_research(
                title="Test",
                content="Content",
            )

            assert content_id is None

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, kas_client):
        """Test health check returns True when KAS is healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(kas_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            is_healthy = await kas_client.health_check()

            assert is_healthy is True
            mock_client.get.assert_called_once_with("/api/v1/health", timeout=5.0)

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, kas_client):
        """Test health check returns False when KAS is down."""
        with patch.object(kas_client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.RequestError("Connection refused")
            )
            mock_get_client.return_value = mock_client

            is_healthy = await kas_client.health_check()

            assert is_healthy is False


class TestGetKAS:
    """Tests for get_kas singleton function."""

    @pytest.fixture(autouse=True)
    async def reset_singleton(self):
        """Reset the KAS singleton before and after each test."""
        await reset_kas()
        yield
        await reset_kas()

    def test_get_kas_disabled(self):
        """Test get_kas returns None when KAS is disabled."""
        with patch("localcrew.integrations.kas.settings") as mock_settings:
            mock_settings.kas_enabled = False
            result = get_kas()
            assert result is None

    def test_get_kas_enabled(self):
        """Test get_kas returns client when KAS is enabled."""
        with patch("localcrew.integrations.kas.settings") as mock_settings:
            mock_settings.kas_enabled = True
            mock_settings.kas_base_url = "http://localhost:8000"
            mock_settings.kas_timeout = 10.0
            mock_settings.kas_api_key = None

            result = get_kas()

            assert result is not None
            assert isinstance(result, KASClient)

    def test_get_kas_singleton(self):
        """Test get_kas returns same instance."""
        with patch("localcrew.integrations.kas.settings") as mock_settings:
            mock_settings.kas_enabled = True
            mock_settings.kas_base_url = "http://localhost:8000"
            mock_settings.kas_timeout = 10.0
            mock_settings.kas_api_key = None

            result1 = get_kas()
            result2 = get_kas()

            assert result1 is result2
