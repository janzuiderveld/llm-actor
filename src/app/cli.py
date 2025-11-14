from __future__ import annotations

import asyncio
from typing import Optional

import typer

from .pipeline_2_personas import run_voice_pipeline

app = typer.Typer(add_completion=False)


@app.command()
def main(session_name: Optional[str] = typer.Option(None, help="Optional session name"), pipeline: str = typer.Option("google", help="Which pipeline to run"),) -> None:
    """Start the Pipecat thin voice pipeline."""
    try:
        if pipeline == "ollama":
            from .pipeline_ollama import run_voice_pipeline
        elif pipeline == "groq":
            from .pipeline_groq import run_voice_pipeline
        else:
            from .pipeline_google import run_voice_pipeline

        asyncio.run(run_voice_pipeline(session_name=session_name))
    except KeyboardInterrupt:  # pragma: no cover - interactive use
        typer.echo("Interrupted by user")


if __name__ == "__main__":
    app()
