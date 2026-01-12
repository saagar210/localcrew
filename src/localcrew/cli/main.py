"""LocalCrew CLI entry point."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from localcrew import __version__
from localcrew.core.config import settings

# API base URL from settings
API_BASE_URL = f"http://localhost:{settings.api_port}/api"

app = typer.Typer(
    name="localcrew",
    help="Local-first multi-agent automation platform",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def callback() -> None:
    """LocalCrew - Local-first multi-agent automation platform."""
    pass


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"[bold blue]LocalCrew[/bold blue] v{__version__}")


@app.command()
def decompose(
    task: str = typer.Argument(..., help="Task description to decompose"),
    project: str | None = typer.Option(None, "--project", "-p", help="Project context"),
    no_sync: bool = typer.Option(False, "--no-sync", help="Don't sync to Task Master"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file (JSON)"),
) -> None:
    """
    Decompose a complex task into actionable subtasks.

    Example:
        localcrew decompose "Add OAuth2 authentication to FastAPI backend"
    """
    import asyncio

    async def run_decompose() -> None:
        import httpx

        console.print(Panel(f"[bold]Decomposing task:[/bold]\n{task}", title="LocalCrew"))

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    f"{API_BASE_URL}/crews/decompose",
                    json={
                        "task": task,
                        "project": project,
                        "auto_sync": not no_sync,
                        "include_taskmaster_context": True,
                    },
                )
                response.raise_for_status()
                result = response.json()

                console.print(f"\n[green]✓[/green] Execution started: {result['execution_id']}")
                console.print(f"Status: {result['status']}")
                console.print(f"\n[dim]Check progress:[/dim] localcrew status {result['execution_id']}")

        except httpx.ConnectError:
            console.print("[red]Error:[/red] Could not connect to LocalCrew API")
            console.print("[dim]Start the server with: uv run fastapi dev src/localcrew/main.py[/dim]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    asyncio.run(run_decompose())


@app.command()
def research(
    query: str = typer.Argument(..., help="Research query"),
    depth: str = typer.Option("medium", "--depth", "-d", help="Research depth: shallow, medium, deep"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file"),
    store_kas: bool = typer.Option(False, "--store-kas", help="Store findings to KAS"),
) -> None:
    """
    Research a topic using the research crew.

    Example:
        localcrew research "Compare CrewAI vs AutoGen"
    """
    import asyncio

    async def run_research() -> None:
        import httpx

        console.print(Panel(f"[bold]Researching:[/bold]\n{query}", title="LocalCrew"))

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    f"{API_BASE_URL}/crews/research",
                    json={
                        "query": query,
                        "depth": depth,
                        "output_format": "markdown",
                        "store_to_kas": store_kas,
                    },
                )
                response.raise_for_status()
                result = response.json()

                console.print(f"\n[green]✓[/green] Research started: {result['execution_id']}")
                console.print(f"Status: {result['status']}")

        except httpx.ConnectError:
            console.print("[red]Error:[/red] Could not connect to LocalCrew API")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    asyncio.run(run_research())


@app.command()
def status(
    execution_id: str | None = typer.Argument(None, help="Execution ID to check"),
    last: int = typer.Option(10, "--last", "-n", help="Show last N executions"),
) -> None:
    """
    Check execution status.

    Example:
        localcrew status
        localcrew status <execution-id>
    """
    import asyncio

    async def check_status() -> None:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                if execution_id:
                    # Get specific execution
                    response = await client.get(
                        f"{API_BASE_URL}/executions/{execution_id}"
                    )
                    response.raise_for_status()
                    execution = response.json()

                    console.print(Panel(
                        f"[bold]ID:[/bold] {execution['id']}\n"
                        f"[bold]Type:[/bold] {execution['crew_type']}\n"
                        f"[bold]Status:[/bold] {execution['status']}\n"
                        f"[bold]Confidence:[/bold] {execution.get('confidence_score', 'N/A')}\n"
                        f"[bold]Duration:[/bold] {execution.get('duration_ms', 'N/A')}ms",
                        title="Execution Details",
                    ))

                    if execution.get("error_message"):
                        console.print(f"[red]Error:[/red] {execution['error_message']}")

                else:
                    # List recent executions
                    response = await client.get(
                        f"{API_BASE_URL}/executions?limit={last}"
                    )
                    response.raise_for_status()
                    executions = response.json()

                    table = Table(title="Recent Executions")
                    table.add_column("ID", style="dim")
                    table.add_column("Type")
                    table.add_column("Status")
                    table.add_column("Confidence")
                    table.add_column("Created")

                    for ex in executions:
                        status_color = {
                            "completed": "green",
                            "failed": "red",
                            "running": "yellow",
                            "pending": "dim",
                            "review_required": "magenta",
                        }.get(ex["status"], "white")

                        table.add_row(
                            ex["id"][:8] + "...",
                            ex["crew_type"],
                            f"[{status_color}]{ex['status']}[/{status_color}]",
                            str(ex.get("confidence_score", "-")),
                            ex["created_at"][:19],
                        )

                    console.print(table)

        except httpx.ConnectError:
            console.print("[red]Error:[/red] Could not connect to LocalCrew API")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    asyncio.run(check_status())


@app.command()
def review(
    review_id: str | None = typer.Argument(None, help="Review ID to process"),
    pending: bool = typer.Option(False, "--pending", "-p", help="Show pending reviews"),
    stats: bool = typer.Option(False, "--stats", "-s", help="Show review statistics"),
) -> None:
    """
    Manage human reviews for low-confidence decompositions.

    Example:
        localcrew review --pending
        localcrew review --stats
        localcrew review <review-id>
    """
    import asyncio

    async def manage_reviews() -> None:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                if stats:
                    # Show review statistics
                    response = await client.get(f"{API_BASE_URL}/reviews/stats")
                    response.raise_for_status()
                    data = response.json()

                    console.print(Panel(
                        f"[bold]Total Reviews:[/bold] {data['total']}\n"
                        f"[bold]Pending:[/bold] {data['pending']}\n"
                        f"[bold]By Decision:[/bold] {data['by_decision']}",
                        title="Review Statistics",
                    ))
                    return

                if pending or not review_id:
                    # List pending reviews
                    response = await client.get(
                        f"{API_BASE_URL}/reviews/pending"
                    )
                    response.raise_for_status()
                    reviews = response.json()

                    if not reviews:
                        console.print("[green]✓[/green] No pending reviews")
                        return

                    table = Table(title="Pending Reviews")
                    table.add_column("ID", style="dim")
                    table.add_column("Confidence")
                    table.add_column("Content Preview")
                    table.add_column("Created")

                    for r in reviews:
                        content = r.get("original_content", {})
                        preview = content.get("title", str(content))[:40]
                        table.add_row(
                            r["id"][:8] + "...",
                            str(r["confidence_score"]),
                            preview + "...",
                            r["created_at"][:19],
                        )

                    console.print(table)
                    console.print(f"\n[dim]Review with: localcrew review <id>[/dim]")

                else:
                    # Show specific review for processing
                    response = await client.get(
                        f"{API_BASE_URL}/reviews/{review_id}"
                    )
                    response.raise_for_status()
                    review_data = response.json()

                    import json
                    console.print(Panel(
                        f"[bold]Confidence:[/bold] {review_data['confidence_score']}%\n\n"
                        f"[bold]Content:[/bold]\n{json.dumps(review_data['original_content'], indent=2)}",
                        title=f"Review {review_data['id'][:8]}...",
                    ))

                    console.print("\n[bold]Actions:[/bold]")
                    console.print("  localcrew approve <id>           - Accept as-is")
                    console.print("  localcrew approve <id> --modify  - Accept with changes")
                    console.print("  localcrew reject <id>            - Reject this item")
                    console.print("  localcrew rerun <id>             - Rerun with guidance")

        except httpx.ConnectError:
            console.print("[red]Error:[/red] Could not connect to LocalCrew API")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    asyncio.run(manage_reviews())


@app.command()
def approve(
    review_id: str = typer.Argument(..., help="Review ID to approve"),
    modify: bool = typer.Option(False, "--modify", "-m", help="Modify content before approval"),
    feedback: str | None = typer.Option(None, "--feedback", "-f", help="Feedback for future improvements"),
    sync: bool = typer.Option(True, "--sync/--no-sync", help="Sync to Task Master after approval"),
) -> None:
    """
    Approve a pending review.

    Example:
        localcrew approve abc123
        localcrew approve abc123 --modify
        localcrew approve abc123 --feedback "Good decomposition"
    """
    import asyncio

    async def do_approve() -> None:
        import httpx
        import json

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Get the review first
                response = await client.get(f"{API_BASE_URL}/reviews/{review_id}")
                response.raise_for_status()
                review_data = response.json()

                modified_content = None
                decision = "approved"

                if modify:
                    # Show content and allow editing
                    console.print(Panel(
                        json.dumps(review_data['original_content'], indent=2),
                        title="Original Content",
                    ))
                    console.print("\n[bold]Enter modified JSON (or press Enter to use original):[/bold]")

                    # For CLI, we'll use a simple prompt
                    # In a real app, this might open an editor
                    user_input = typer.prompt("Modified JSON", default="")
                    if user_input.strip():
                        try:
                            modified_content = json.loads(user_input)
                            decision = "modified"
                        except json.JSONDecodeError:
                            console.print("[red]Invalid JSON. Using original content.[/red]")

                # Submit the review
                response = await client.post(
                    f"{API_BASE_URL}/reviews/{review_id}/submit",
                    json={
                        "decision": decision,
                        "modified_content": modified_content,
                        "feedback": feedback,
                    },
                )
                response.raise_for_status()

                console.print(f"[green]✓[/green] Review {decision}")

                if sync:
                    # Trigger Task Master sync
                    response = await client.post(
                        f"{API_BASE_URL}/reviews/{review_id}/sync"
                    )
                    if response.status_code == 200:
                        console.print("[green]✓[/green] Synced to Task Master")
                    else:
                        console.print("[yellow]![/yellow] Task Master sync skipped")

        except httpx.ConnectError:
            console.print("[red]Error:[/red] Could not connect to LocalCrew API")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    asyncio.run(do_approve())


@app.command()
def reject(
    review_id: str = typer.Argument(..., help="Review ID to reject"),
    feedback: str | None = typer.Option(None, "--feedback", "-f", help="Reason for rejection"),
) -> None:
    """
    Reject a pending review.

    Example:
        localcrew reject abc123
        localcrew reject abc123 --feedback "Task too vague"
    """
    import asyncio

    async def do_reject() -> None:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{API_BASE_URL}/reviews/{review_id}/submit",
                    json={
                        "decision": "rejected",
                        "feedback": feedback,
                    },
                )
                response.raise_for_status()
                console.print(f"[yellow]✗[/yellow] Review rejected")

        except httpx.ConnectError:
            console.print("[red]Error:[/red] Could not connect to LocalCrew API")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    asyncio.run(do_reject())


@app.command()
def rerun(
    review_id: str = typer.Argument(..., help="Review ID to rerun"),
    guidance: str = typer.Option(..., "--guidance", "-g", help="Guidance for the rerun"),
) -> None:
    """
    Rerun a decomposition with additional guidance.

    Example:
        localcrew rerun abc123 --guidance "Focus more on testing steps"
    """
    import asyncio

    async def do_rerun() -> None:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Submit as rerun
                response = await client.post(
                    f"{API_BASE_URL}/reviews/{review_id}/submit",
                    json={
                        "decision": "rerun",
                        "feedback": guidance,
                    },
                )
                response.raise_for_status()

                # Trigger the rerun
                response = await client.post(
                    f"{API_BASE_URL}/reviews/{review_id}/rerun",
                    json={"guidance": guidance},
                )

                if response.status_code == 200:
                    result = response.json()
                    console.print(f"[green]✓[/green] Rerun started: {result.get('execution_id', 'unknown')}")
                else:
                    console.print("[yellow]![/yellow] Rerun queued for processing")

        except httpx.ConnectError:
            console.print("[red]Error:[/red] Could not connect to LocalCrew API")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    asyncio.run(do_rerun())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(settings.api_port, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
) -> None:
    """
    Start the LocalCrew API server.

    Example:
        localcrew serve
        localcrew serve --port 8080 --reload
    """
    import uvicorn

    console.print(f"[bold blue]Starting LocalCrew API server...[/bold blue]")
    console.print(f"Host: {host}:{port}")

    uvicorn.run(
        "localcrew.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
