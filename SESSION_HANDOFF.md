# LocalCrew - Session Handoff
**Last Updated:** January 12, 2026
**Branch:** `main`
**Status:** ✅ Production Ready (local-first)

---

## Project Overview

LocalCrew is a local-first multi-agent automation platform for intelligent task decomposition and research. Built on CrewAI, runs 100% on Apple Silicon using MLX.

**Tech Stack:**
- Backend: FastAPI 0.128+, SQLModel, PostgreSQL 16+
- AI: CrewAI 1.8.0+, MLX 0.30+, Qwen2.5:14B-Q4
- Frontend: Next.js 15, shadcn/ui
- Integrations: Task Master AI (MCP), KAS (optional), MLflow (optional)

---

## Current Status

### ✅ Completed Features

1. **Task Decomposition Crew**
   - Breaks complex tasks into subtasks
   - Confidence scoring (0-100)
   - Human review gate for confidence < 70%
   - Structured outputs via outlines library

2. **Research Crew**
   - Multi-agent research pipeline (decomposer → gatherer → synthesizer → reporter)
   - Structured JSON outputs enforced by Pydantic schemas
   - Query decomposition → information gathering → synthesis → report generation

3. **KAS Integration** (Optional)
   - Pre-query personal knowledge base before external research
   - Auto-ingest research reports back to KAS
   - Sync/async HTTP methods (uvloop compatible)
   - Health check shows connection status
   - Dashboard widget shows KB stats

4. **Human Review System**
   - Review queue for low-confidence outputs
   - Approve/modify/reject/rerun workflows
   - Feedback storage for prompt improvement
   - Task Master sync for approved decompositions

5. **Dashboard MVP**
   - Home: stats + recent executions
   - Executions: history with filtering
   - Reviews: queue with actions
   - Workflows: decomposition + research

6. **Security Hardening**
   - Input validation (sanitize control characters)
   - KAS API key validation (prevent header injection)
   - Production deployment checklist in CLAUDE.md
   - Full .env.example with security notes

### 🔄 Configured But Not Active

1. **MLflow Tracking**
   - Configuration in place
   - Not logging metrics yet
   - Need to add tracking calls in services

2. **Feedback Loop**
   - Collecting feedback from reviews
   - Not analyzing or applying improvements yet
   - Database tables exist: `feedback` table

---

## Recent Accomplishments (Jan 11-12, 2026)

### Session 1: KAS Integration
- Implemented KAS client with sync/async methods
- Fixed uvloop compatibility issue
- Added 28 KAS-specific tests
- Integrated KAS into research crew
- Auto-ingest research reports to KAS

### Session 2: Security Audit & Fixes
- Comprehensive security audit (15 issues identified)
- Fixed critical issues:
  - Task Master content validation
  - KAS API key sanitization
  - UTC timestamp deprecation
- Updated documentation (.env.example, CLAUDE.md)
- All 51 tests passing

### Final State
- Merged PR to main: 2,528 additions, 190 deletions
- Branch deleted: `feat/structured-outputs-integration`
- Clean main branch, ready for next feature

---

## Architecture Overview

```
src/localcrew/
├── api/
│   └── routes/          # FastAPI endpoints (crews, reviews, executions, health)
├── agents/              # Agent definitions (planner, validator, gatherer, etc.)
├── crews/               # CrewAI flows (decomposition, research)
├── models/              # SQLModel database models
├── services/            # Business logic (decomposition, research)
├── integrations/        # External services
│   ├── kas.py          # Knowledge Activation System
│   ├── mlx.py          # MLX inference wrapper
│   ├── structured_mlx.py          # Outlines + MLX
│   ├── structured_crewai_llm.py   # CrewAI LLM adapter
│   └── taskmaster.py   # Task Master MCP integration
├── schemas/             # Pydantic schemas for structured outputs
└── cli/                # Typer CLI commands

web/
├── src/
│   ├── app/            # Next.js pages
│   └── components/     # React components (dashboard, reviews, executions)
└── public/

tests/
├── test_kas.py         # KAS client tests
├── test_research_kas.py # KAS integration tests
└── test_structured_outputs.py # Schema validation tests
```

---

## Key Decisions & Design Patterns

### 1. Local-First Architecture
- No cloud dependencies
- Default configs for localhost
- CORS: localhost ports 3000-3002
- DB: Default credentials (localcrew/localcrew)
- Production deployment optional (see CLAUDE.md)

### 2. Graceful Degradation
- All integrations optional and fault-tolerant
- KAS unavailable → research continues without KB
- Task Master unavailable → decomposition works standalone
- MLflow unavailable → no metrics, execution continues

### 3. Structured Outputs
- Use outlines library for guaranteed JSON
- All agent outputs validated by Pydantic
- Schemas in `src/localcrew/schemas/agent_outputs.py`
- Eliminates JSON parsing errors

### 4. Human Review Gate
- Confidence threshold: 70% (configurable)
- Below threshold → review queue
- Above threshold → auto-approved
- Feedback stored for future prompt optimization

### 5. KAS Integration Philosophy
- Pre-query for existing knowledge (score > 0.7)
- Auto-ingest all research reports
- Separate "From your knowledge base" sources in reports
- Use sync HTTP for CrewAI flows (uvloop compatibility)

---

## Configuration

### Environment Variables (.env)

```bash
# Required
DATABASE_URL=postgresql+asyncpg://localcrew:localcrew@localhost:5432/localcrew

# MLX
MLX_MODEL_ID=mlx-community/Qwen2.5-14B-Instruct-4bit
MLX_FALLBACK_MODEL_ID=mlx-community/Qwen2.5-7B-Instruct-4bit

# Optional: KAS
KAS_ENABLED=false
KAS_BASE_URL=http://localhost:8000
KAS_API_KEY=optional
KAS_TIMEOUT=10.0

# Optional: MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=localcrew
```

See `.env.example` for full reference.

---

## Testing

```bash
# Run all tests (51 tests)
uv run pytest tests/ -v

# Test coverage by area:
# - Health checks: 2 tests
# - KAS client: 13 tests
# - KAS integration: 14 tests
# - Structured outputs: 18 tests
```

All tests passing as of Jan 12, 2026.

---

## Next Steps (Prioritized Roadmap)

### 1. Feedback Loop System (High Priority)
**Why:** Already collecting feedback, not using it yet.

**Implementation:**
- Analyze feedback data for common patterns
- Auto-generate prompt improvements
- A/B test prompts, track confidence changes
- Store best-performing prompts per domain

**Files to create:**
- `src/localcrew/services/prompt_optimizer.py`
- `src/localcrew/api/routes/prompt_optimization.py`

### 2. MLflow Metrics Tracking (Quick Win)
**Why:** Config exists, just need to add logging calls.

**Implementation:**
- Log execution time, token usage, confidence scores
- Track model performance (14B vs 7B fallback)
- Compare crew types (decomposition vs research)
- Dashboard visualizations

**Files to modify:**
- `src/localcrew/services/base.py` (add mlflow logging)
- `src/localcrew/services/decomposition.py`
- `src/localcrew/services/research.py`

### 3. New Crew Types (Expand Use Cases)
**Options:**
- Code review crew (analyze PRs)
- Documentation crew (generate/update docs)
- Bug analysis crew (investigate issues)
- Refactoring crew (identify code smells)

### 4. Advanced KAS Features
- Query KAS during decomposition (not just research)
- Store validated patterns automatically
- Use KAS for agent memory/context
- Domain-specific pattern libraries

### 5. Dashboard Real-Time Updates
- WebSocket for live execution progress
- Streaming output as agents work
- Live confidence scores
- Inline approval UI

---

## Known Issues & Limitations

### Current Limitations
1. **No Authentication** - Designed for local network only
2. **No Rate Limiting** - Add slowapi for production
3. **Single User** - No multi-tenant support
4. **MLX Only** - Requires Apple Silicon (M1/M2/M3/M4)

### Deferred Features
1. **Prompt Optimization** - Feedback collected but not analyzed
2. **MLflow Logging** - Config exists, calls not added
3. **Agent Memory** - No persistent memory across sessions
4. **Retry Logic** - Executions endpoint has retry stub, not implemented

### Production Considerations
See `CLAUDE.md` → Security Notes → Production Deployment Checklist:
- Change DB credentials
- Update CORS origins
- Add rate limiting
- Enable HTTPS (reverse proxy)
- Consider authentication

---

## Development Workflow

```bash
# Start backend
uv run fastapi dev src/localcrew/main.py

# Start frontend (from web/)
npm run dev

# Run tests
uv run pytest tests/ -v

# Format code
uv run ruff format src/ tests/
uv run ruff check --fix src/ tests/

# CLI commands
localcrew decompose "task description"
localcrew research "query"
localcrew review --pending
```

---

## Git Status

**Branch:** `main`
**Last Commit:** `de37859` (security hardening)
**Remote:** `origin/main` (up to date)

**Recent Commits:**
```
de37859 security: add input validation and production deployment docs
874784e feat: integrate Knowledge Activation System (KAS)
85c65ab feat: integrate outlines for structured agent outputs
ddc4211 fix: expand CORS to allow ports 3001 and 3002
```

**Uncommitted Changes:** None (on main)
**Untracked Files:** `.taskmaster/tasks/` (Task Master working directory)

---

## Important Files

### Documentation
- `CLAUDE.md` - Project instructions + security notes
- `.env.example` - Full configuration reference
- `SESSION_HANDOFF.md` - This file (session state)

### Configuration
- `pyproject.toml` - Dependencies + Python config
- `uv.lock` - Locked dependencies
- `.env` - Environment variables (gitignored)

### Key Source Files
- `src/localcrew/main.py` - FastAPI application entry
- `src/localcrew/crews/research.py` - Research flow with KAS
- `src/localcrew/integrations/kas.py` - KAS client
- `src/localcrew/services/research.py` - Research service with auto-ingest
- `src/localcrew/api/routes/reviews.py` - Review endpoints with validation

---

## Context for Next Session

### What Just Happened
- Completed major KAS integration
- Fixed all critical security issues
- Merged feature branch to main
- All tests passing, clean state

### What's Ready to Build
1. **Feedback loop** - Database and models exist, just need analysis logic
2. **MLflow metrics** - Config ready, add tracking calls
3. **New crew type** - Template exists, easy to add more

### What to Know
- KAS uses **sync HTTP client** in CrewAI flows (uvloop compatibility)
- All agent outputs use **structured schemas** (outlines library)
- **Human review** triggered by confidence < 70%
- **Security audit** identified remaining items (see CLAUDE.md)

### Questions to Resolve
1. Which crew type to build next?
2. Should we prioritize feedback loop or MLflow first?
3. Do we need Task Master initialization for this project?

---

## Quick Start for New Session

```bash
# 1. Check git status
git status
git log --oneline -5

# 2. Run tests
uv run pytest tests/ -v

# 3. Start dev servers
uv run fastapi dev src/localcrew/main.py  # Backend on :8001
cd web && npm run dev                      # Frontend on :3000

# 4. Check what's next
# See "Next Steps" section above
```

---

**Session End:** January 12, 2026
**Next Session:** Ready to implement feedback loop or MLflow metrics
**Status:** ✅ Clean, tested, documented, merged to main
