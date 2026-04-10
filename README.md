# Project Kinetic

A ComfyUI workflow builder for cinematic anime video generation (Ufotable/MAPPA aesthetic). Targets 10–30 second sequences at 24fps with 4K output on an NVIDIA RTX 4070 Ti Super (16GB VRAM).

## Overview

Project Kinetic builds [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflow JSON in Python and submits it to a running ComfyUI server via its HTTP/WebSocket API. The pipeline is implemented in stages — each stage produces a complete, submittable workflow.

**Current status:** Stage 1 (Multi-Modal Motion Extraction) implemented.

## Pipeline Stages

### Stage 1 — Multi-Modal Motion Extraction

Extracts motion control signals from a source video across three parallel buses:

| Bus      | Nodes                                            | Purpose                                                               |
| -------- | ------------------------------------------------ | --------------------------------------------------------------------- |
| Organic  | DWPose_Estimator + DensePose                     | Skeletal/facial landmarking + volumetric depth for human choreography |
| Rigid    | ControlNet_LineArt_Anime + Canny_Edge + ZoeDepth | Edge geometry + metric depth for vehicle/hard-surface dynamics        |
| Temporal | Unimatch_Optical_Flow                            | Motion vectors (pixel displacement t→t+1) for temporal lock           |

### Stages 2–4 — (planned)

- Stage 2: Temporal Latent Engine — Wan 2.6 (NVFP4), IP-Adapter FaceID Plus, FreeLong spectral blending
- Stage 3: Dynamic VFX — SAM 3 tracked masking, elemental inpainting sub-routine
- Stage 4: High-Resolution Synthesis — Tiled VAE decode, RTX VSR 4K upscaling

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- A running [ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance with the required custom nodes installed

### Required ComfyUI Custom Nodes (Stage 1)

- `VHS_LoadVideo` — [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- `DWPose_Estimator`, `DensePose` — [comfyui-controlnet-aux](https://github.com/Fannovel16/comfyui_controlnet_aux)
- `ControlNet_LineArt_Anime`, `Canny_Edge` — [comfyui-controlnet-aux](https://github.com/Fannovel16/comfyui_controlnet_aux)
- `ZoeDepth` — [comfyui-controlnet-aux](https://github.com/Fannovel16/comfyui_controlnet_aux)
- `Unimatch_Optical_Flow` — [ComfyUI-UniMatch](https://github.com/dajes/ComfyUI-UniMatch)

## Installation

```bash
git clone <repo-url>
cd anime-vid-generator
uv sync
```

For development (includes pytest, respx, etc.):

```bash
uv sync --dev
```

## Usage

### CLI

```bash
# Preview the Stage 1 workflow JSON without submitting
uv run kinetic stage1 /path/to/source.mp4 --dry-run

# Submit Stage 1 to a running ComfyUI instance
uv run kinetic stage1 /path/to/source.mp4

# Target a non-default ComfyUI server
uv run kinetic stage1 /path/to/source.mp4 --comfyui-url http://192.168.1.10:8188
```

Or via `main.py`:

```bash
uv run python main.py stage1 /path/to/source.mp4 --dry-run
```

### Python API

```python
from anime_vid_generator.workflow.stages.stage1_motion import build_stage1_workflow
from anime_vid_generator.config import Stage1Config

# Build workflow JSON with defaults
workflow = build_stage1_workflow("/path/to/source.mp4")

# Build with custom config
config = Stage1Config(
    context_window_frames=16,
    context_overlap_frames=4,
    canny_low_threshold=80,
    canny_high_threshold=180,
    zoe_depth_model="ZoeD_K",
)
workflow = build_stage1_workflow("/path/to/source.mp4", config=config)

# Submit to ComfyUI
import asyncio
from anime_vid_generator.client import ComfyUIClient

async def run():
    client = ComfyUIClient("http://127.0.0.1:8188")
    prompt_id = await client.submit_workflow(workflow)
    history = await client.get_history(prompt_id)

asyncio.run(run())
```

## Configuration

All config is pydantic-based with sensible defaults matching the TDD hardware targets.

### `HardwareConfig`

| Field                    | Default                 | Description                  |
| ------------------------ | ----------------------- | ---------------------------- |
| `vram_gb`                | `16.0`                  | Total VRAM envelope (GB)     |
| `base_model_vram_gb`     | `4.2`                   | Wan 2.6 NVFP4 footprint      |
| `controlnet_vram_gb`     | `2.0`                   | ControlNet stack budget      |
| `context_window_vram_gb` | `5.5`                   | 32-frame latent chunk budget |
| `buffer_vram_gb`         | `4.3`                   | PyTorch overhead buffer      |
| `comfyui_url`            | `http://127.0.0.1:8188` | ComfyUI server address       |

### `Stage1Config`

| Field                     | Default   | Description                                            |
| ------------------------- | --------- | ------------------------------------------------------ |
| `context_window_frames`   | `32`      | Frames per temporal chunk                              |
| `context_overlap_frames`  | `8`       | Frame overlap between chunks                           |
| `fps`                     | `24`      | Target frame rate                                      |
| `optical_flow_raft_iters` | `12`      | Unimatch RAFT iterations                               |
| `zoe_depth_model`         | `ZoeD_NK` | ZoeDepth model variant (`ZoeD_N`, `ZoeD_K`, `ZoeD_NK`) |
| `canny_low_threshold`     | `100`     | Canny edge lower threshold                             |
| `canny_high_threshold`    | `200`     | Canny edge upper threshold                             |

## Development

```bash
# Run tests (no ComfyUI required)
uv run pytest

# Run tests with output
uv run pytest -v
```

All 58 unit tests run without a live ComfyUI instance or GPU.

## Project Structure

```
src/anime_vid_generator/
├── config.py                     # HardwareConfig, Stage1Config, PipelineConfig
├── client.py                     # ComfyUIClient (HTTP + WebSocket)
├── cli.py                        # Typer CLI (kinetic command)
└── workflow/
    ├── nodes.py                  # WorkflowNode dataclass, node factory functions
    ├── builder.py                # WorkflowBuilder (auto-ID, JSON serialization)
    └── stages/
        └── stage1_motion.py      # Organic bus, rigid bus, temporal unification
```
