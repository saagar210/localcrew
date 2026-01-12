# LocalCrew - Project Instructions

## Overview
LocalCrew is a local-first multi-agent automation platform for intelligent task decomposition. Built on CrewAI, runs 100% on Apple Silicon using MLX.

## Tech Stack
- **Backend:** FastAPI 0.128+, SQLModel, PostgreSQL 16+
- **AI:** CrewAI 1.8.0+, MLX 0.30+, Qwen2.5:14B-Q4
- **CLI:** Typer
- **Dashboard:** Next.js 15, shadcn/ui
- **Metrics:** MLflow 3.8+

## Key Commands
```bash
# Development
uv run fastapi dev src/localcrew/main.py
uv run pytest tests/

# CLI (after install)
localcrew decompose "task description"
localcrew research "query"
localcrew review --pending
```

## Project Structure
```
src/localcrew/
├── api/           # FastAPI routes
├── crews/         # CrewAI crew definitions
├── agents/        # Individual agent configs
├── models/        # SQLModel database models
├── services/      # Business logic
├── integrations/  # Task Master, MLflow
└── cli/           # Typer CLI commands
```

## Integration Points
- **Task Master AI:** Read/write tasks via MCP tools
- **MLX:** Direct inference, no Ollama wrapper
- **MLflow:** Agent performance tracking

## Development Rules
- MLX-native inference only (no Ollama)
- Human review gate for confidence < 70%
- All crews return structured output with confidence scores
- Standalone service (separate from KAS)

## Current Phase
Phase 5: Dashboard MVP (completed)
- Next.js 15 with shadcn/ui
- Home dashboard with stats and recent executions
- Executions history view with filtering
- Reviews queue with approve/reject/rerun actions
- Workflows view for task decomposition and research

## Dashboard Commands
```bash
# Start dashboard (from web/ directory)
npm run dev      # Development server on port 3000
npm run build    # Production build
```

## Completed Phases
1. Foundation: FastAPI, PostgreSQL, CLI
2. CrewAI Integration: Flows, MLX wrapper, Task Master sync
3. Human Review: Review queue, feedback storage, CLI commands
4. Research Crew: Multi-agent research pipeline
5. Dashboard MVP: Next.js frontend
