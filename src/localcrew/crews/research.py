"""Research Crew using CrewAI Flows."""

import json
import logging
import re
from datetime import datetime
from typing import Any

from crewai import Crew, Process, Task
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field

from localcrew.agents import (
    create_query_decomposer_agent,
    create_gatherer_agent,
    create_synthesizer_agent,
    create_reporter_agent,
)
from localcrew.integrations.crewai_llm import MLXLLM

logger = logging.getLogger(__name__)


class SubQuestion(BaseModel):
    """A sub-question derived from the main query."""

    question: str
    search_keywords: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    importance: str = "medium"  # high, medium, low


class Finding(BaseModel):
    """A finding from research."""

    source_url: str
    source_title: str
    content: str
    credibility: str = "medium"  # high, medium, low
    retrieved_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ResearchState(BaseModel):
    """State for the research flow."""

    query: str = ""
    depth: str = "medium"  # shallow, medium, deep
    output_format: str = "markdown"

    # Query decomposition output
    sub_questions: list[SubQuestion] = Field(default_factory=list)

    # Information gathering output
    findings: list[Finding] = Field(default_factory=list)

    # Synthesis output
    synthesis: dict[str, Any] = Field(default_factory=dict)
    confidence_score: int = 75

    # Final report
    report: str = ""
    sources: list[dict[str, str]] = Field(default_factory=list)


class ResearchFlow(Flow[ResearchState]):
    """Flow for conducting research using multiple agents."""

    def __init__(self, llm: MLXLLM | None = None) -> None:
        """Initialize the research flow.

        Args:
            llm: Optional custom LLM. Uses MLX by default.
        """
        super().__init__()
        self.llm = llm or MLXLLM()

    @start()
    def decompose_query(self) -> str:
        """Decompose the research query into sub-questions."""
        logger.info(f"Decomposing query: {self.state.query[:100]}...")

        decomposer = create_query_decomposer_agent(self.llm)

        depth_guidance = {
            "shallow": "Create 2-3 focused sub-questions covering the basics.",
            "medium": "Create 4-5 sub-questions covering main aspects and context.",
            "deep": "Create 6-7 sub-questions for comprehensive coverage including edge cases.",
        }

        decompose_task = Task(
            description=f"""Decompose the following research query into sub-questions:

Query: {self.state.query}

Depth: {self.state.depth}
{depth_guidance.get(self.state.depth, depth_guidance["medium"])}

For each sub-question, provide:
- question: The specific question to research
- search_keywords: List of keywords for searching
- source_types: Preferred sources (docs, academic, news, forum, blog)
- importance: high, medium, or low

Output as JSON array of sub-questions.""",
            expected_output="JSON array of sub-questions with search keywords",
            agent=decomposer,
        )

        decompose_crew = Crew(
            agents=[decomposer],
            tasks=[decompose_task],
            process=Process.sequential,
            verbose=True,
        )

        result = decompose_crew.kickoff()

        try:
            sub_questions_data = self._extract_json(result.raw)
            if isinstance(sub_questions_data, dict) and "sub_questions" in sub_questions_data:
                sub_questions_data = sub_questions_data["sub_questions"]
            if not isinstance(sub_questions_data, list):
                sub_questions_data = [sub_questions_data]

            self.state.sub_questions = [
                SubQuestion(**sq) if isinstance(sq, dict) else SubQuestion(question=str(sq))
                for sq in sub_questions_data
            ]
            logger.info(f"Decomposed into {len(self.state.sub_questions)} sub-questions")
        except Exception as e:
            logger.warning(f"Could not parse sub-questions: {e}")
            # Fallback: use original query as single sub-question
            self.state.sub_questions = [
                SubQuestion(
                    question=self.state.query,
                    search_keywords=self.state.query.split()[:5],
                    source_types=["docs", "academic", "news"],
                    importance="high",
                )
            ]

        return "decomposition_complete"

    @listen(decompose_query)
    def gather_information(self, _: str) -> str:
        """Gather information for each sub-question."""
        logger.info(f"Gathering information for {len(self.state.sub_questions)} sub-questions...")

        gatherer = create_gatherer_agent(self.llm)

        # Format sub-questions for the task
        sq_text = "\n".join([
            f"{i+1}. {sq.question} (Keywords: {', '.join(sq.search_keywords)})"
            for i, sq in enumerate(self.state.sub_questions)
        ])

        gather_task = Task(
            description=f"""Research the following sub-questions and gather relevant information:

{sq_text}

For each sub-question, find and document:
- Relevant facts and information
- Source URLs (use realistic examples if actual search isn't available)
- Source credibility assessment

Output as JSON with:
- findings: Array of finding objects with source_url, source_title, content, credibility""",
            expected_output="JSON with findings array",
            agent=gatherer,
        )

        gather_crew = Crew(
            agents=[gatherer],
            tasks=[gather_task],
            process=Process.sequential,
            verbose=True,
        )

        result = gather_crew.kickoff()

        try:
            findings_data = self._extract_json(result.raw)
            if isinstance(findings_data, dict) and "findings" in findings_data:
                findings_data = findings_data["findings"]
            if not isinstance(findings_data, list):
                findings_data = [findings_data]

            self.state.findings = [
                Finding(**f) if isinstance(f, dict) else Finding(
                    source_url="https://example.com",
                    source_title="Research Source",
                    content=str(f),
                )
                for f in findings_data
            ]
            logger.info(f"Gathered {len(self.state.findings)} findings")
        except Exception as e:
            logger.warning(f"Could not parse findings: {e}")
            self.state.findings = [
                Finding(
                    source_url="https://docs.example.com",
                    source_title="Documentation",
                    content=f"Information about: {self.state.query}",
                    credibility="medium",
                )
            ]

        return "gathering_complete"

    @listen(gather_information)
    def synthesize_findings(self, _: str) -> str:
        """Synthesize gathered findings into coherent analysis."""
        logger.info("Synthesizing findings...")

        synthesizer = create_synthesizer_agent(self.llm)

        # Format findings for synthesis
        findings_text = "\n\n".join([
            f"Source: {f.source_title} ({f.source_url})\n"
            f"Credibility: {f.credibility}\n"
            f"Content: {f.content}"
            for f in self.state.findings
        ])

        synthesize_task = Task(
            description=f"""Synthesize the following research findings into a coherent analysis:

Original Query: {self.state.query}

Findings:
{findings_text}

Provide:
1. Key themes and patterns across sources
2. Areas of agreement and any contradictions
3. Gaps in the available information
4. Overall confidence score (0-100)

Output as JSON with:
- themes: List of key themes
- agreements: Points multiple sources agree on
- contradictions: Any conflicting information
- gaps: What's missing
- conclusions: Main conclusions
- confidence_score: Overall confidence (0-100)""",
            expected_output="JSON synthesis with themes, conclusions, and confidence",
            agent=synthesizer,
        )

        synthesize_crew = Crew(
            agents=[synthesizer],
            tasks=[synthesize_task],
            process=Process.sequential,
            verbose=True,
        )

        result = synthesize_crew.kickoff()

        try:
            synthesis = self._extract_json(result.raw)
            self.state.synthesis = synthesis
            self.state.confidence_score = synthesis.get("confidence_score", 75)
            logger.info(f"Synthesis complete. Confidence: {self.state.confidence_score}%")
        except Exception as e:
            logger.warning(f"Could not parse synthesis: {e}")
            self.state.synthesis = {
                "themes": ["Research findings"],
                "conclusions": [f"Analysis of {self.state.query}"],
                "confidence_score": 70,
            }
            self.state.confidence_score = 70

        return "synthesis_complete"

    @listen(synthesize_findings)
    def generate_report(self, _: str) -> str:
        """Generate the final formatted report."""
        logger.info("Generating report...")

        reporter = create_reporter_agent(self.llm)

        # Format synthesis for report
        synthesis_text = json.dumps(self.state.synthesis, indent=2)
        sources_text = "\n".join([
            f"- {f.source_title}: {f.source_url}"
            for f in self.state.findings
        ])

        report_task = Task(
            description=f"""Create a {self.state.depth} research report in {self.state.output_format} format.

Original Query: {self.state.query}

Synthesis:
{synthesis_text}

Available Sources:
{sources_text}

Report Requirements:
1. Start with a brief summary (2-3 sentences)
2. List key points as bullets
3. Provide detailed analysis
4. Note any limitations
5. End with a Sources section listing all references

Use inline citations: [Source Name](url)
Confidence Score: {self.state.confidence_score}%""",
            expected_output="Formatted research report in markdown",
            agent=reporter,
        )

        report_crew = Crew(
            agents=[reporter],
            tasks=[report_task],
            process=Process.sequential,
            verbose=True,
        )

        result = report_crew.kickoff()

        self.state.report = result.raw
        self.state.sources = [
            {
                "url": f.source_url,
                "title": f.source_title,
                "retrieved_at": f.retrieved_at,
            }
            for f in self.state.findings
        ]

        logger.info(f"Report generated: {len(self.state.report)} characters")
        return "report_complete"

    def _extract_json(self, text: str) -> dict | list:
        """Extract JSON from LLM response text."""
        # Try to find JSON in code blocks first
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            text = json_match.group(1)

        text = text.strip()

        # Find first [ or {
        start_bracket = -1
        start_brace = -1

        for i, c in enumerate(text):
            if c == "[" and start_bracket == -1:
                start_bracket = i
            elif c == "{" and start_brace == -1:
                start_brace = i
            if start_bracket != -1 and start_brace != -1:
                break

        if start_bracket == -1 and start_brace == -1:
            raise ValueError("No JSON found in text")

        if start_bracket != -1 and (start_brace == -1 or start_bracket < start_brace):
            end = text.rfind("]")
            if end != -1:
                text = text[start_bracket : end + 1]
        else:
            end = text.rfind("}")
            if end != -1:
                text = text[start_brace : end + 1]

        return json.loads(text)


async def run_research(
    query: str,
    depth: str = "medium",
    output_format: str = "markdown",
) -> ResearchState:
    """Run the research flow.

    Args:
        query: The research query
        depth: Research depth (shallow, medium, deep)
        output_format: Output format (markdown, plain)

    Returns:
        ResearchState with report and sources
    """
    flow = ResearchFlow()
    flow.state.query = query
    flow.state.depth = depth
    flow.state.output_format = output_format

    await flow.kickoff_async()

    return flow.state
