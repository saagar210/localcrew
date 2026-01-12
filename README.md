# LocalCrew - AI Task Automation Platform

A local-first multi-agent automation platform for intelligent task decomposition and research. Built on CrewAI, runs 100% on Apple Silicon using MLX.

## Overview

LocalCrew provides:
- **Task Decomposition**: Break complex tasks into actionable subtasks using AI agents
- **Research Crew**: Deep research with multi-agent pipeline (decompose, gather, synthesize, report)
- **Human Review**: Review low-confidence results before syncing to Task Master
- **Task Master Integration**: Sync decomposed tasks to your project's Task Master tasks.json

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI 0.128+, SQLModel, PostgreSQL 16+ |
| AI | CrewAI 1.8.0+, MLX 0.30+, Qwen2.5:14B-Q4 |
| CLI | Typer |
| Dashboard | Next.js 15, shadcn/ui |
| Metrics | MLflow 3.8+ |

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 16+
- Node.js 20+ (for dashboard)
- [uv](https://github.com/astral-sh/uv) package manager

### 1. Clone and Setup

```bash
cd ~/claude-code/personal/crewai-automation-platform

# Install Python dependencies
uv sync

# Copy environment file
cp .env.example .env
# Edit .env with your database credentials
```

### 2. Setup Database

```bash
# Start PostgreSQL (if using Docker)
docker run -d \
  --name localcrew-db \
  -e POSTGRES_USER=localcrew \
  -e POSTGRES_PASSWORD=localcrew \
  -e POSTGRES_DB=localcrew \
  -p 5433:5432 \
  postgres:16

# Tables are auto-created on first run
```

### 3. Start Backend

```bash
# Development mode with auto-reload
uv run fastapi dev src/localcrew/main.py --port 8001

# Production mode
uv run fastapi run src/localcrew/main.py --port 8001
```

API available at http://localhost:8001/api

### 4. Start Dashboard (Optional)

```bash
cd web
npm install
npm run dev
```

Dashboard available at http://localhost:3000

## Architecture

```
src/localcrew/
├── api/              # FastAPI routes
│   └── routes/       # health, crews, executions, reviews, workflows
├── agents/           # CrewAI agent definitions
│   ├── analyzer.py       # Task context analyzer
│   ├── planner.py        # Subtask planner
│   ├── validator.py      # Output validator
│   ├── query_decomposer.py  # Research query decomposer
│   ├── gatherer.py       # Information gatherer
│   ├── synthesizer.py    # Finding synthesizer
│   └── reporter.py       # Report generator
├── crews/            # CrewAI Flow definitions
│   ├── decomposition.py  # Task decomposition workflow
│   └── research.py       # Research workflow
├── models/           # SQLModel database models
├── services/         # Business logic layer
├── integrations/     # External integrations
│   ├── mlx.py           # MLX inference wrapper
│   ├── crewai_llm.py    # CrewAI LLM adapter
│   └── taskmaster.py    # Task Master sync
├── core/             # Core utilities
│   ├── config.py        # Settings via pydantic-settings
│   ├── database.py      # Async database setup
│   └── types.py         # Custom SQLAlchemy types
└── cli/              # Typer CLI
```

## API Endpoints

### Health
- `GET /api/health` - Liveness check
- `GET /api/health/ready` - Readiness check

### Crews
- `GET /api/crews/types` - List available crew types
- `POST /api/crews/decompose` - Decompose a task
- `POST /api/crews/research` - Run research query

### Executions
- `GET /api/executions` - List executions with filters
- `GET /api/executions/{id}` - Get execution details
- `GET /api/executions/{id}/subtasks` - Get execution subtasks
- `POST /api/executions/{id}/retry` - Retry failed execution

### Reviews
- `GET /api/reviews/pending` - List pending reviews
- `GET /api/reviews/stats` - Review statistics
- `GET /api/reviews/{id}` - Get review details
- `POST /api/reviews/{id}/submit` - Submit review decision
- `POST /api/reviews/{id}/sync` - Sync to Task Master
- `POST /api/reviews/{id}/rerun` - Rerun with guidance

### Workflows
- `GET /api/workflows` - List workflows
- `POST /api/workflows` - Create workflow
- `GET /api/workflows/{id}` - Get workflow
- `DELETE /api/workflows/{id}` - Deactivate workflow

## CLI Usage

```bash
# Install CLI
uv pip install -e .

# Decompose a task
localcrew decompose "Implement user authentication with JWT"

# Research a topic
localcrew research "Best practices for Python async" --depth deep

# View pending reviews
localcrew review --pending

# Approve/reject reviews
localcrew approve <review-id>
localcrew reject <review-id> --reason "Needs more detail"
```

## Human Review System

Tasks with confidence < 70% are flagged for human review:

1. **Pending**: AI generated output awaiting review
2. **Approved**: Output accepted as-is
3. **Modified**: Output edited before sync
4. **Rejected**: Output rejected
5. **Rerun**: Rerun with additional guidance

Review decisions are stored for prompt improvement analysis.

## Task Master Integration

Decomposed subtasks can be synced to your project's Task Master:

```bash
# Auto-sync enabled by default
localcrew decompose "Build API endpoint" --project my-project

# Manual sync after review
localcrew sync <review-id>
```

Tasks are written to `.taskmaster/tasks/tasks.json`.

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=localcrew

# Verbose output
uv run pytest -v
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint
uv run ruff check .

# Type check
uv run mypy src/localcrew
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | postgresql+asyncpg://... | PostgreSQL connection |
| MLX_MODEL | qwen2.5:14b-q4 | MLX model for inference |
| CONFIDENCE_THRESHOLD | 70 | Review threshold (0-100) |
| MLFLOW_TRACKING_URI | http://localhost:5001 | MLflow server |

## Completed Phases

1. **Foundation** - FastAPI, PostgreSQL, SQLModel, Typer CLI
2. **CrewAI Integration** - Flows, MLX wrapper, agent definitions
3. **Human Review** - Review queue, feedback storage, CLI commands
4. **Research Crew** - Multi-agent research pipeline
5. **Dashboard MVP** - Next.js frontend with shadcn/ui

## License

MIT
