from __future__ import annotations

import asyncio
from typing import Optional

import typer

#from .pipeline import run_voice_pipeline
from .pipeline_2_personas import run_voice_pipeline

app = typer.Typer(add_completion=False)


@app.command()
def main(session_name: Optional[str] = typer.Option(None, help="Optional session name")) -> None:
    """Start the Pipecat thin voice pipeline."""
    try:
        asyncio.run(run_voice_pipeline(session_name=session_name))
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        typer.echo("Interrupted by user")


if __name__ == "__main__":
    app()
