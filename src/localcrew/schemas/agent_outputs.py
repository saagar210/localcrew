"""Pydantic schemas for structured agent outputs.

These schemas define the expected output format for each agent in the
decomposition and research crews. Using outlines with these schemas
guarantees valid JSON output that matches the schema exactly.
"""

from typing import Literal

from pydantic import BaseModel, Field


# Decomposition Crew Schemas
class SubtaskItem(BaseModel):
    """A single subtask in a decomposition plan."""

    title: str = Field(description="Clear, concise task title in imperative form")
    description: str = Field(description="Detailed description of what needs to be done")
    subtask_type: Literal["coding", "research", "devops", "documentation", "design", "testing"] = Field(
        description="Type of subtask"
    )
    estimated_complexity: Literal["low", "medium", "high"] = Field(
        description="Estimated complexity level"
    )
    dependencies: list[int] = Field(
        default_factory=list, description="Indices of subtasks this depends on (0-indexed)"
    )
    order_index: int = Field(description="Suggested execution order (0, 1, 2, ...)")


class TaskAnalysis(BaseModel):
    """Output from the Analyzer agent."""

    domain: Literal["coding", "research", "devops", "documentation", "design", "testing"] = Field(
        description="Primary domain of the task"
    )
    complexity: Literal["low", "medium", "high"] = Field(
        description="Overall task complexity"
    )
    explicit_requirements: list[str] = Field(
        description="Clearly stated requirements from the task"
    )
    implicit_requirements: list[str] = Field(
        default_factory=list, description="Inferred requirements not explicitly stated"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="External dependencies or prerequisites"
    )
    risks: list[str] = Field(
        default_factory=list, description="Potential issues or blockers"
    )
    estimated_subtask_count: int = Field(
        ge=3, le=10, description="Estimated number of subtasks (3-10)"
    )


class SubtaskPlan(BaseModel):
    """Output from the Planner agent - a wrapper for list of subtasks."""

    subtasks: list[SubtaskItem] = Field(description="List of subtasks to complete")


class ValidatedSubtask(BaseModel):
    """A subtask with validation confidence score added."""

    title: str = Field(description="Clear, concise task title")
    description: str = Field(description="Detailed description of what needs to be done")
    subtask_type: Literal["coding", "research", "devops", "documentation", "design", "testing"] = Field(
        description="Type of subtask"
    )
    estimated_complexity: Literal["low", "medium", "high"] = Field(
        description="Estimated complexity level"
    )
    dependencies: list[int] = Field(
        default_factory=list, description="Indices of subtasks this depends on"
    )
    order_index: int = Field(description="Execution order")
    confidence_score: int = Field(
        ge=0, le=100, description="Confidence score (0-100)"
    )


class ValidationResult(BaseModel):
    """Output from the Validator agent."""

    validated_subtasks: list[ValidatedSubtask] = Field(
        description="Subtasks with confidence scores added"
    )
    overall_confidence: int = Field(
        ge=0, le=100, description="Average confidence score (0-100)"
    )
    issues: list[str] = Field(
        default_factory=list, description="Concerns found in the plan"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Improvements if any"
    )


# Research Crew Schemas
class SubQuestion(BaseModel):
    """A sub-question for research."""

    question: str = Field(description="The specific question to research")
    search_keywords: list[str] = Field(
        default_factory=list, description="Keywords for searching"
    )
    source_types: list[str] = Field(
        default_factory=list, description="Preferred sources (docs, academic, news, forum, blog)"
    )
    importance: Literal["high", "medium", "low"] = Field(
        default="medium", description="Importance level"
    )


class QueryDecomposition(BaseModel):
    """Output from the Query Decomposer agent."""

    sub_questions: list[SubQuestion] = Field(description="Decomposed sub-questions")


class FindingItem(BaseModel):
    """A single research finding."""

    source_url: str = Field(description="URL of the source")
    source_title: str = Field(description="Title of the source")
    content: str = Field(description="The finding content")
    credibility: Literal["high", "medium", "low"] = Field(
        default="medium", description="Source credibility"
    )


class GatheredFindings(BaseModel):
    """Output from the Research Gatherer agent."""

    findings: list[FindingItem] = Field(description="Research findings collected")


class ResearchSynthesis(BaseModel):
    """Output from the Research Synthesizer agent."""

    themes: list[str] = Field(description="Key themes across sources")
    agreements: list[str] = Field(
        default_factory=list, description="Points multiple sources agree on"
    )
    contradictions: list[str] = Field(
        default_factory=list, description="Conflicting information"
    )
    gaps: list[str] = Field(
        default_factory=list, description="Missing information"
    )
    conclusions: list[str] = Field(description="Main conclusions")
    confidence_score: int = Field(
        ge=0, le=100, description="Overall confidence (0-100)"
    )
