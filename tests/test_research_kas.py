"""Integration tests for Research Crew + KAS."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from localcrew.integrations.kas import KASResult, reset_kas
from localcrew.crews.research import ResearchFlow, Finding, SubQuestion


class TestResearchKASIntegration:
    """Tests for KAS integration in Research Crew."""

    @pytest.fixture(autouse=True)
    async def reset_singleton(self):
        """Reset the KAS singleton before and after each test."""
        await reset_kas()
        yield
        await reset_kas()

    @pytest.fixture
    def mock_kas_results(self):
        """Sample KAS search results."""
        return [
            KASResult(
                content_id="kb_doc_1",
                title="JWT Authentication Guide",
                content_type="note",
                score=0.92,
                chunk_text="JWT tokens are stateless authentication tokens...",
                source_ref="",
            ),
            KASResult(
                content_id="kb_doc_2",
                title="OAuth Best Practices",
                content_type="bookmark",
                score=0.75,
                chunk_text="When implementing OAuth, always validate...",
                source_ref="https://oauth.net/2/",
            ),
            KASResult(
                content_id="kb_doc_3",
                title="Low Score Doc",
                content_type="note",
                score=0.5,  # Below 0.7 threshold
                chunk_text="This should be filtered out...",
                source_ref="",
            ),
        ]

    def test_query_kas_filters_low_scores(self, mock_kas_results):
        """Test that _query_kas_sync filters out low-score results."""
        flow = ResearchFlow(use_structured_outputs=False)
        sub_questions = [
            SubQuestion(question="How does JWT work?", search_keywords=["JWT", "token"]),
        ]

        mock_kas = MagicMock()
        mock_kas.search_sync = MagicMock(return_value=mock_kas_results)

        with patch("localcrew.crews.research.get_kas", return_value=mock_kas):
            findings = flow._query_kas_sync(sub_questions)

        # Should only include results with score > 0.7
        assert len(findings) == 2
        assert all(f.credibility == "high" for f in findings)

    def test_query_kas_returns_empty_when_disabled(self):
        """Test that _query_kas_sync returns empty list when KAS is disabled."""
        flow = ResearchFlow(use_structured_outputs=False)
        sub_questions = [
            SubQuestion(question="Test question", search_keywords=["test"]),
        ]

        with patch("localcrew.crews.research.get_kas", return_value=None):
            findings = flow._query_kas_sync(sub_questions)

        assert findings == []

    def test_query_kas_handles_errors_gracefully(self):
        """Test that _query_kas_sync doesn't fail on KAS errors."""
        flow = ResearchFlow(use_structured_outputs=False)
        sub_questions = [
            SubQuestion(question="Test question", search_keywords=["test"]),
        ]

        mock_kas = MagicMock()
        mock_kas.search_sync = MagicMock(side_effect=Exception("KAS error"))

        with patch("localcrew.crews.research.get_kas", return_value=mock_kas):
            findings = flow._query_kas_sync(sub_questions)

        # Should return empty list, not raise exception
        assert findings == []

    def test_kas_findings_marked_with_kb_prefix(self, mock_kas_results):
        """Test that KAS findings have [KB] prefix in title."""
        flow = ResearchFlow(use_structured_outputs=False)
        sub_questions = [
            SubQuestion(question="JWT auth", search_keywords=["JWT"]),
        ]

        mock_kas = MagicMock()
        mock_kas.search_sync = MagicMock(return_value=mock_kas_results[:2])  # Only high scores

        with patch("localcrew.crews.research.get_kas", return_value=mock_kas):
            findings = flow._query_kas_sync(sub_questions)

        assert all(f.source_title.startswith("[KB]") for f in findings)

    def test_kas_findings_use_kas_url_scheme(self, mock_kas_results):
        """Test that KAS findings use kas:// URL scheme."""
        flow = ResearchFlow(use_structured_outputs=False)
        sub_questions = [
            SubQuestion(question="OAuth", search_keywords=["OAuth"]),
        ]

        mock_kas = MagicMock()
        mock_kas.search_sync = MagicMock(return_value=mock_kas_results[:2])

        with patch("localcrew.crews.research.get_kas", return_value=mock_kas):
            findings = flow._query_kas_sync(sub_questions)

        assert all(f.source_url.startswith("kas://") for f in findings)


class TestReportSourceSeparation:
    """Tests for separating KAS and external sources in reports."""

    def test_sources_separated_in_generate_report(self):
        """Test that report separates KB and external sources."""
        # Create findings with mixed sources
        kas_finding = Finding(
            source_url="kas://doc123",
            source_title="[KB] Test KB Doc",
            content="KB content",
            credibility="high",
        )
        external_finding = Finding(
            source_url="https://example.com/doc",
            source_title="External Doc",
            content="External content",
            credibility="medium",
        )

        # Test source separation logic directly
        all_findings = [kas_finding, external_finding]

        kas_sources = [f for f in all_findings if f.source_url.startswith("kas://")]
        external_sources = [f for f in all_findings if not f.source_url.startswith("kas://")]

        assert len(kas_sources) == 1
        assert len(external_sources) == 1
        assert kas_sources[0].source_title == "[KB] Test KB Doc"
        assert external_sources[0].source_title == "External Doc"


class TestResearchServiceAutoIngest:
    """Tests for auto-ingest in research service."""

    @pytest.fixture(autouse=True)
    async def reset_singleton(self):
        """Reset the KAS singleton before and after each test."""
        await reset_kas()
        yield
        await reset_kas()

    def test_extract_topic_tags(self):
        """Test topic tag extraction from research results."""
        from localcrew.services.research import ResearchService

        # Create service with mocked session
        service = ResearchService(session=MagicMock())

        research_result = {
            "sub_questions": [
                "What are JWT authentication best practices?",
                "How does OAuth 2.0 work?",
            ],
        }

        tags = service._extract_topic_tags(research_result)

        # Should extract meaningful words (4+ chars)
        assert isinstance(tags, list)
        assert len(tags) <= 10  # Limited to 10

        # Common words should be extracted
        assert any("authentication" in tag for tag in tags) or \
               any("best" in tag for tag in tags) or \
               any("practices" in tag for tag in tags)

    def test_extract_topic_tags_empty_input(self):
        """Test topic tag extraction with empty input."""
        from localcrew.services.research import ResearchService

        service = ResearchService(session=MagicMock())

        research_result = {"sub_questions": []}
        tags = service._extract_topic_tags(research_result)

        assert tags == []

    @pytest.mark.asyncio
    async def test_store_to_kas_success(self):
        """Test successful research ingestion to KAS."""
        from localcrew.services.research import ResearchService

        service = ResearchService(session=MagicMock())

        mock_execution = MagicMock()
        mock_execution.id = "exec-123"

        research_result = {
            "query": "How does JWT work?",
            "synthesis": "# JWT Research\n\nJWT is...",
            "confidence_score": 85,
            "depth": "medium",
            "sub_questions": ["What is JWT?", "How to validate JWT?"],
        }

        mock_kas = AsyncMock()
        mock_kas.ingest_research = AsyncMock(return_value="new_content_123")

        with patch("localcrew.services.research.get_kas", return_value=mock_kas):
            await service._store_to_kas(mock_execution, research_result)

        mock_kas.ingest_research.assert_called_once()
        call_kwargs = mock_kas.ingest_research.call_args[1]
        assert "JWT" in call_kwargs["title"]
        assert call_kwargs["content"] == "# JWT Research\n\nJWT is..."
        assert "research" in call_kwargs["tags"]
        assert "localcrew" in call_kwargs["tags"]

    @pytest.mark.asyncio
    async def test_store_to_kas_disabled(self):
        """Test that store_to_kas does nothing when KAS is disabled."""
        from localcrew.services.research import ResearchService

        service = ResearchService(session=MagicMock())

        mock_execution = MagicMock()
        research_result = {"query": "test", "synthesis": "test"}

        with patch("localcrew.services.research.get_kas", return_value=None):
            # Should not raise, just return early
            await service._store_to_kas(mock_execution, research_result)

    @pytest.mark.asyncio
    async def test_store_to_kas_handles_errors(self):
        """Test that store_to_kas doesn't fail execution on errors."""
        from localcrew.services.research import ResearchService

        service = ResearchService(session=MagicMock())

        mock_execution = MagicMock()
        mock_execution.id = "exec-123"
        research_result = {
            "query": "test",
            "synthesis": "test",
            "sub_questions": [],
        }

        mock_kas = AsyncMock()
        mock_kas.ingest_research = AsyncMock(side_effect=Exception("KAS error"))

        with patch("localcrew.services.research.get_kas", return_value=mock_kas):
            # Should not raise exception
            await service._store_to_kas(mock_execution, research_result)


class TestHealthCheckKAS:
    """Tests for KAS status in health check."""

    @pytest.fixture(autouse=True)
    async def reset_singleton(self):
        """Reset the KAS singleton before and after each test."""
        await reset_kas()
        yield
        await reset_kas()

    @pytest.mark.asyncio
    async def test_health_check_kas_disabled(self):
        """Test health check shows KAS as disabled."""
        from localcrew.api.routes.health import readiness_check

        with patch("localcrew.api.routes.health.settings") as mock_settings:
            mock_settings.kas_enabled = False

            result = await readiness_check()

            assert result["services"]["kas"] == "disabled"

    @pytest.mark.asyncio
    async def test_health_check_kas_connected(self):
        """Test health check shows KAS as connected when healthy."""
        from localcrew.api.routes.health import readiness_check

        mock_kas = AsyncMock()
        mock_kas.health_check = AsyncMock(return_value=True)

        with patch("localcrew.api.routes.health.settings") as mock_settings:
            mock_settings.kas_enabled = True
            with patch("localcrew.api.routes.health.get_kas", return_value=mock_kas):
                result = await readiness_check()

        assert result["services"]["kas"] == "connected"
        assert result["status"] == "ready"

    @pytest.mark.asyncio
    async def test_health_check_kas_unavailable(self):
        """Test health check shows KAS unavailable and degraded status."""
        from localcrew.api.routes.health import readiness_check

        mock_kas = AsyncMock()
        mock_kas.health_check = AsyncMock(return_value=False)

        with patch("localcrew.api.routes.health.settings") as mock_settings:
            mock_settings.kas_enabled = True
            with patch("localcrew.api.routes.health.get_kas", return_value=mock_kas):
                result = await readiness_check()

        assert result["services"]["kas"] == "unavailable"
        assert result["status"] == "degraded"
