"""Tests for crew endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_crew_types(client: AsyncClient) -> None:
    """Test listing available crew types."""
    response = await client.get("/api/crews/types")
    assert response.status_code == 200

    data = response.json()
    assert "crews" in data
    assert len(data["crews"]) >= 2

    crew_types = [c["type"] for c in data["crews"]]
    assert "task_decomposition" in crew_types
    assert "research" in crew_types


@pytest.mark.asyncio
async def test_decompose_task_validation(client: AsyncClient) -> None:
    """Test task decomposition input validation."""
    # Task too short
    response = await client.post(
        "/api/crews/decompose",
        json={"task": "short"},
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_research_query_validation(client: AsyncClient) -> None:
    """Test research query input validation."""
    # Query too short
    response = await client.post(
        "/api/crews/research",
        json={"query": "hi"},
    )
    assert response.status_code == 422  # Validation error

    # Invalid depth
    response = await client.post(
        "/api/crews/research",
        json={"query": "Valid research query", "depth": "invalid"},
    )
    assert response.status_code == 422
