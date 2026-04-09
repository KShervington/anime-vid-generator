import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .client import ComfyUIClient
from .config import PipelineConfig
from .workflow.stages.stage1_motion import build_stage1_workflow

app = typer.Typer(name="kinetic", help="Project Kinetic: Anime video generation pipeline")
console = Console()


@app.callback()
def main_callback() -> None:
    """Project Kinetic: Anime video generation pipeline."""


@app.command()
def stage1(
    video: Path = typer.Argument(..., help="Path to source video file", exists=True),
    comfyui_url: str = typer.Option("http://127.0.0.1:8188", help="ComfyUI server URL"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print workflow JSON without submitting"),
) -> None:
    """Run Stage 1: Multi-Modal Motion Extraction on a source video."""
    config = PipelineConfig()
    config.hardware.comfyui_url = comfyui_url
    workflow = build_stage1_workflow(str(video.resolve()), config.stage1)

    if dry_run:
        console.print_json(json.dumps(workflow))
        raise typer.Exit()

    asyncio.run(_submit_and_monitor(workflow, config))


async def _submit_and_monitor(workflow: dict, config: PipelineConfig) -> None:
    client = ComfyUIClient(config.hardware.comfyui_url)
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Submitting Stage 1 workflow...", total=None)
        prompt_id = await client.submit_workflow(workflow)
        progress.update(task, description=f"Running (ID: {prompt_id})...")
        async for event in client.monitor_progress(prompt_id):
            if event.get("type") == "progress":
                data = event.get("data", {})
                progress.update(
                    task,
                    description=f"Node {data.get('node', '?')}: {data.get('value', 0)}/{data.get('max', 0)}",
                )
    console.print(f"[green]Stage 1 complete. Prompt ID: {prompt_id}[/green]")
