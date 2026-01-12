"""Tests for structured output schemas and generators."""

import pytest
from pydantic import ValidationError

from localcrew.schemas import (
    # Decomposition schemas
    SubtaskItem,
    TaskAnalysis,
    SubtaskPlan,
    ValidatedSubtask,
    ValidationResult,
    # Research schemas
    SubQuestion,
    QueryDecomposition,
    FindingItem,
    GatheredFindings,
    ResearchSynthesis,
)


class TestDecompositionSchemas:
    """Tests for decomposition crew schemas."""

    def test_subtask_item_valid(self):
        """Test SubtaskItem with valid data."""
        subtask = SubtaskItem(
            title="Implement authentication",
            description="Add JWT-based authentication to the API",
            subtask_type="coding",
            estimated_complexity="medium",
            dependencies=[0, 1],
            order_index=2,
        )
        assert subtask.title == "Implement authentication"
        assert subtask.subtask_type == "coding"
        assert subtask.estimated_complexity == "medium"

    def test_subtask_item_invalid_type(self):
        """Test SubtaskItem rejects invalid subtask_type."""
        with pytest.raises(ValidationError):
            SubtaskItem(
                title="Test",
                description="Test description",
                subtask_type="invalid_type",  # Invalid
                estimated_complexity="medium",
                order_index=0,
            )

    def test_subtask_item_invalid_complexity(self):
        """Test SubtaskItem rejects invalid complexity."""
        with pytest.raises(ValidationError):
            SubtaskItem(
                title="Test",
                description="Test description",
                subtask_type="coding",
                estimated_complexity="very_high",  # Invalid
                order_index=0,
            )

    def test_task_analysis_valid(self):
        """Test TaskAnalysis with valid data."""
        analysis = TaskAnalysis(
            domain="coding",
            complexity="high",
            explicit_requirements=["User authentication", "JWT tokens"],
            implicit_requirements=["Secure password storage"],
            dependencies=["Database setup"],
            risks=["Security vulnerabilities"],
            estimated_subtask_count=5,
        )
        assert analysis.domain == "coding"
        assert analysis.estimated_subtask_count == 5

    def test_task_analysis_subtask_count_bounds(self):
        """Test TaskAnalysis enforces subtask count bounds."""
        with pytest.raises(ValidationError):
            TaskAnalysis(
                domain="coding",
                complexity="medium",
                explicit_requirements=["Test"],
                estimated_subtask_count=2,  # Less than 3
            )

        with pytest.raises(ValidationError):
            TaskAnalysis(
                domain="coding",
                complexity="medium",
                explicit_requirements=["Test"],
                estimated_subtask_count=15,  # More than 10
            )

    def test_subtask_plan_valid(self):
        """Test SubtaskPlan with valid data."""
        plan = SubtaskPlan(
            subtasks=[
                SubtaskItem(
                    title="Research",
                    description="Research auth patterns",
                    subtask_type="research",
                    estimated_complexity="low",
                    order_index=0,
                ),
                SubtaskItem(
                    title="Implement",
                    description="Implement auth",
                    subtask_type="coding",
                    estimated_complexity="high",
                    dependencies=[0],
                    order_index=1,
                ),
            ]
        )
        assert len(plan.subtasks) == 2

    def test_validation_result_valid(self):
        """Test ValidationResult with valid data."""
        result = ValidationResult(
            validated_subtasks=[
                ValidatedSubtask(
                    title="Test task",
                    description="Test description",
                    subtask_type="testing",
                    estimated_complexity="low",
                    order_index=0,
                    confidence_score=85,
                )
            ],
            overall_confidence=85,
            issues=["Minor dependency issue"],
            suggestions=["Consider adding more tests"],
        )
        assert result.overall_confidence == 85
        assert len(result.validated_subtasks) == 1

    def test_validation_result_confidence_bounds(self):
        """Test ValidationResult enforces confidence bounds."""
        with pytest.raises(ValidationError):
            ValidationResult(
                validated_subtasks=[],
                overall_confidence=150,  # Over 100
            )

        with pytest.raises(ValidationError):
            ValidationResult(
                validated_subtasks=[],
                overall_confidence=-10,  # Negative
            )


class TestResearchSchemas:
    """Tests for research crew schemas."""

    def test_sub_question_valid(self):
        """Test SubQuestion with valid data."""
        sq = SubQuestion(
            question="What are the best practices for JWT authentication?",
            search_keywords=["JWT", "authentication", "best practices"],
            source_types=["docs", "academic"],
            importance="high",
        )
        assert sq.importance == "high"
        assert len(sq.search_keywords) == 3

    def test_query_decomposition_valid(self):
        """Test QueryDecomposition with valid data."""
        qd = QueryDecomposition(
            sub_questions=[
                SubQuestion(
                    question="What is JWT?",
                    search_keywords=["JWT", "JSON Web Token"],
                    importance="high",
                ),
                SubQuestion(
                    question="How to implement JWT in Python?",
                    search_keywords=["JWT", "Python", "implementation"],
                    importance="medium",
                ),
            ]
        )
        assert len(qd.sub_questions) == 2

    def test_finding_item_valid(self):
        """Test FindingItem with valid data."""
        finding = FindingItem(
            source_url="https://jwt.io/introduction",
            source_title="JWT Introduction",
            content="JWT is a compact, URL-safe means of representing claims...",
            credibility="high",
        )
        assert finding.credibility == "high"

    def test_finding_item_invalid_credibility(self):
        """Test FindingItem rejects invalid credibility."""
        with pytest.raises(ValidationError):
            FindingItem(
                source_url="https://example.com",
                source_title="Test",
                content="Test content",
                credibility="very_high",  # Invalid
            )

    def test_gathered_findings_valid(self):
        """Test GatheredFindings with valid data."""
        findings = GatheredFindings(
            findings=[
                FindingItem(
                    source_url="https://example.com",
                    source_title="Example",
                    content="Example content",
                    credibility="medium",
                )
            ]
        )
        assert len(findings.findings) == 1

    def test_research_synthesis_valid(self):
        """Test ResearchSynthesis with valid data."""
        synthesis = ResearchSynthesis(
            themes=["Security", "Performance", "Scalability"],
            agreements=["JWT is stateless"],
            contradictions=["Token expiry times vary by source"],
            gaps=["Limited mobile implementation details"],
            conclusions=["JWT is suitable for API authentication"],
            confidence_score=80,
        )
        assert synthesis.confidence_score == 80
        assert len(synthesis.themes) == 3

    def test_research_synthesis_confidence_bounds(self):
        """Test ResearchSynthesis enforces confidence bounds."""
        with pytest.raises(ValidationError):
            ResearchSynthesis(
                themes=["Test"],
                conclusions=["Test"],
                confidence_score=101,  # Over 100
            )


class TestSchemaJsonSerialization:
    """Tests for JSON serialization of schemas."""

    def test_subtask_item_to_json(self):
        """Test SubtaskItem serializes to JSON correctly."""
        subtask = SubtaskItem(
            title="Test",
            description="Test desc",
            subtask_type="coding",
            estimated_complexity="low",
            order_index=0,
        )
        json_data = subtask.model_dump()
        assert json_data["title"] == "Test"
        assert json_data["subtask_type"] == "coding"
        assert json_data["dependencies"] == []

    def test_task_analysis_from_json(self):
        """Test TaskAnalysis can be created from JSON."""
        json_data = {
            "domain": "research",
            "complexity": "low",
            "explicit_requirements": ["Find information about X"],
            "implicit_requirements": [],
            "dependencies": [],
            "risks": [],
            "estimated_subtask_count": 3,
        }
        analysis = TaskAnalysis(**json_data)
        assert analysis.domain == "research"

    def test_validation_result_roundtrip(self):
        """Test ValidationResult survives JSON roundtrip."""
        import json

        result = ValidationResult(
            validated_subtasks=[
                ValidatedSubtask(
                    title="Task 1",
                    description="Description",
                    subtask_type="coding",
                    estimated_complexity="medium",
                    order_index=0,
                    confidence_score=90,
                )
            ],
            overall_confidence=90,
            issues=[],
            suggestions=[],
        )

        # Serialize to JSON string and back
        json_str = json.dumps(result.model_dump())
        parsed = json.loads(json_str)
        restored = ValidationResult(**parsed)

        assert restored.overall_confidence == 90
        assert restored.validated_subtasks[0].title == "Task 1"
