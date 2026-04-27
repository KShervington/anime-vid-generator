import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .client import ComfyUIClient
from .config import PipelineConfig, Stage2Config, Stage3Config
from .workflow.stages.stage1_motion import build_stage1_workflow
from .workflow.stages.stage2_generation import build_stage2_workflow
from .workflow.stages.stage3_vfx import build_stage3_workflow

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


@app.command()
def stage2(
    video: Path = typer.Argument(..., help="Path to source video file", exists=True),
    reference_image: str = typer.Option("", "--reference-image", help="Path to face reference image"),
    prompt: str = typer.Option(
        "cinematic anime, ufotable style, high quality",
        "--prompt",
        help="Positive text prompt",
    ),
    negative_prompt: str = typer.Option(
        "blurry, low quality, photorealistic",
        "--negative-prompt",
        help="Negative text prompt",
    ),
    vae_encode: bool = typer.Option(False, "--vae-encode", help="Use VAEEncode instead of EmptyLatentVideo"),
    denoise: float = typer.Option(1.0, "--denoise", help="KSampler denoise strength"),
    comfyui_url: str = typer.Option("http://127.0.0.1:8188", help="ComfyUI server URL"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print workflow JSON without submitting"),
) -> None:
    """Run Stage 2: Temporal Latent Engine — model loading, identity anchoring, and generation."""
    pipeline_config = PipelineConfig()
    pipeline_config.hardware.comfyui_url = comfyui_url

    config = Stage2Config(
        reference_image_path=reference_image,
        positive_prompt=prompt,
        negative_prompt=negative_prompt,
        latent_mode="vae_encode" if vae_encode else "empty",
        denoise=denoise,
    )
    workflow = build_stage2_workflow(str(video.resolve()), config=config)

    if dry_run:
        console.print_json(json.dumps(workflow))
        raise typer.Exit()

    asyncio.run(_submit_and_monitor(workflow, pipeline_config))


@app.command()
def stage3(
    video: Path = typer.Argument(..., help="Path to source video file", exists=True),
    emitter_prompt: str = typer.Option("sword tip", "--emitter-prompt", help="SAM 3 emitter description"),
    vfx_lora: str = typer.Option(
        "Ufotable_Fire_Trails.safetensors",
        "--vfx-lora",
        help="VFX LoRA filename",
    ),
    vfx_cfg: float = typer.Option(8.5, "--vfx-cfg", help="CFG scale for VFX inpainting KSampler"),
    vfx_prompt: str = typer.Option(
        "fire trails, vfx, ufotable style, elemental effects",
        "--vfx-prompt",
        help="Positive prompt for VFX inpainting pass",
    ),
    seed: int = typer.Option(0, "--seed", help="Sampler seed"),
    comfyui_url: str = typer.Option("http://127.0.0.1:8188", help="ComfyUI server URL"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print workflow JSON without submitting"),
) -> None:
    """Run Stage 3: Dynamic VFX & Masked Inpainting — emitter tracking and elemental effects."""
    pipeline_config = PipelineConfig()
    pipeline_config.hardware.comfyui_url = comfyui_url

    config = Stage3Config(
        emitter_prompt=emitter_prompt,
        vfx_lora_path=vfx_lora,
        vfx_cfg=vfx_cfg,
        vfx_positive_prompt=vfx_prompt,
        seed=seed,
    )
    workflow = build_stage3_workflow(str(video.resolve()), config)

    if dry_run:
        console.print_json(json.dumps(workflow))
        raise typer.Exit()

    asyncio.run(_submit_and_monitor(workflow, pipeline_config))


async def _submit_and_monitor(workflow: dict, config: PipelineConfig) -> None:
    client = ComfyUIClient(config.hardware.comfyui_url)
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        task = progress.add_task("Submitting workflow...", total=None)
        prompt_id = await client.submit_workflow(workflow)
        progress.update(task, description=f"Running (ID: {prompt_id})...")
        async for event in client.monitor_progress(prompt_id):
            if event.get("type") == "progress":
                data = event.get("data", {})
                progress.update(
                    task,
                    description=f"Node {data.get('node', '?')}: {data.get('value', 0)}/{data.get('max', 0)}",
                )
    console.print(f"[green]Stage complete. Prompt ID: {prompt_id}[/green]")
