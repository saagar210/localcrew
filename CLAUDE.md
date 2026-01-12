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
6. KAS Integration: Personal knowledge base integration

## KAS Integration
Optional integration with Knowledge Activation System for personal knowledge base.

```bash
# Enable KAS
KAS_ENABLED=true
KAS_BASE_URL=http://localhost:8000
KAS_TIMEOUT=10.0
```

Features:
- Pre-query KAS before external research for existing knowledge
- Auto-ingest research reports back to KAS
- Separate KB sources from external sources in reports
- Health check shows KAS connection status

## Security Notes

### Local-First Design
LocalCrew is designed for **local network use only**. The default configuration prioritizes developer experience over production security:

- **CORS**: Allows localhost origins (ports 3000-3002)
- **Database**: Default credentials for easy setup
- **Authentication**: No auth middleware (trusted local network)

### Production Deployment Checklist
If deploying beyond localhost:

1. **Database Credentials**
   ```bash
   # Generate strong password
   openssl rand -base64 32
   # Use SSL: add ?sslmode=require to DATABASE_URL
   ```

2. **CORS Configuration**
   Update `src/localcrew/main.py`:
   ```python
   allow_origins=["https://your-domain.com"]
   ```

3. **Rate Limiting**
   Consider adding [slowapi](https://github.com/laurentS/slowapi):
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   ```

4. **HTTPS**
   Use reverse proxy (nginx/caddy) with TLS certificates

5. **API Authentication**
   Add auth middleware if exposing beyond local network

### Input Validation
- Research queries: 5-2000 characters, sanitized
- Task Master sync: Title/description validated, control characters removed
- KAS API key: Header injection prevention
- All Pydantic models enforce type safety

### Graceful Degradation
All external integrations fail gracefully:
- KAS unavailable: Research continues without KB
- Task Master unavailable: Decomposition works standalone
- MLflow unavailable: No metrics, but execution continues
