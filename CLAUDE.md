# Project Kinetic — CLAUDE.md

## Project Overview

Python 3.13 ComfyUI API wrapper for anime video generation. Builds ComfyUI workflow JSON in Python and submits it to a running ComfyUI server. The pipeline is implemented incrementally — one stage per plan, each stage producing testable software independently.

**Current state:** Stage 1 (Multi-Modal Motion Extraction) and Stage 2 (Temporal Latent Engine) are complete with 167 passing tests.

## Key Architecture Decisions

**Node model:** `WorkflowNode` is a dataclass with a `dict[str, NodeInput]` for inputs. `NodeRef = tuple[str, int]` is used for node-to-node links. `to_api_dict()` serializes tuples to lists for JSON. Never wrap scalars in a container class — ComfyUI takes them bare.

**Auto-assigned IDs:** `WorkflowBuilder.add(node)` returns the auto-assigned string ID ("1", "2", ...). Bus functions receive a `WorkflowBuilder` and `NodeRef` for the source, register their nodes, and return typed result dataclasses with named `NodeRef` fields — no raw ID strings passed around by callers.

**Stages return result dataclasses:** `OrganicBusResult`, `RigidBusResult`, `TemporalUnificationResult`, `ModelLoadingBusResult`, `IdentityBusResult`, `ConditioningBusResult`, `LatentBusResult`, `GenerationBusResult` — downstream stages wire specific outputs by field name, not positional tuple indices.

**All config is pydantic:** `HardwareConfig`, `Stage1Config`, `Stage2Config`, `PipelineConfig`. Defaults match the TDD document exactly (16GB envelope, 32-frame context window, 8-frame overlap, 24fps).

## File Map

```
src/anime_vid_generator/
├── config.py                     # HardwareConfig, Stage1Config, Stage2Config, PipelineConfig
├── client.py                     # ComfyUIClient — HTTP + WebSocket
├── cli.py                        # Typer CLI (kinetic stage1, kinetic stage2)
└── workflow/
    ├── nodes.py                  # WorkflowNode, NodeRef, factory functions (Stage 1 + Stage 2)
    ├── builder.py                # WorkflowBuilder
    └── stages/
        ├── stage1_motion.py      # build_organic_bus, build_rigid_bus,
        │                         # build_temporal_unification, build_stage1_workflow
        └── stage2_generation.py  # build_model_loading_bus, build_conditioning_bus,
                                  # build_identity_bus, build_latent_bus,
                                  # build_generation_bus, build_stage2_workflow
tests/
├── conftest.py                   # builder fixture
├── test_config.py
├── test_client.py
├── test_cli.py
└── workflow/
    ├── test_nodes.py
    ├── test_builder.py
    └── stages/
        ├── test_stage1_motion.py
        └── test_stage2_generation.py
```

## Development Commands

```bash
uv sync --dev          # install all deps including dev
uv run pytest          # run all 167 tests (no ComfyUI needed)
uv run pytest -v       # verbose output
uv run kinetic stage1 <video> --dry-run                      # preview Stage 1 workflow JSON
uv run kinetic stage1 <video>                                # submit Stage 1 to ComfyUI at localhost:8188
uv run kinetic stage2 <video> --dry-run                      # preview Stage 2 workflow JSON
uv run kinetic stage2 <video> --reference-image <face.png>   # submit Stage 2 with identity reference
uv run kinetic stage2 <video> --vae-encode --denoise 0.75    # vid2vid mode at 75% denoise
```

## Testing Rules

- All unit tests must pass without a live ComfyUI instance or GPU
- HTTP interactions: mock with `respx` (already in dev deps)
- WebSocket interactions: mock with `unittest.mock` async context managers
- Integration tests (real server): mark with `@pytest.mark.integration` and skip by default
- `asyncio_mode = "auto"` is set — `async def test_...` just works, no decorator needed

## Adding a New Stage

1. Create a new plan: `docs/superpowers/plans/YYYY-MM-DD-stage-N.md`
2. Add node factories to `workflow/nodes.py` for any new ComfyUI node types
3. Create `workflow/stages/stageN_*.py` following the bus-function + result-dataclass pattern
4. All three buses in Stage 1 take `(builder, image_ref, config)` and return a typed result — follow this pattern
5. Top-level assembler function signature: `build_stageN_workflow(video_path: str, config: StageNConfig | None = None) -> dict`

## ComfyUI API Format

The `build()` output matches the ComfyUI `/prompt` API directly:

```json
{
  "1": {"class_type": "VHS_LoadVideo", "inputs": {"video": "/path/to/file.mp4", ...}},
  "2": {"class_type": "DWPose_Estimator", "inputs": {"image": ["1", 0], ...}}
}
```

- Node IDs are strings
- Node links: `["source_id_string", output_slot_index]` (tuples in Python, arrays in JSON)
- Scalar inputs: bare values

## Stage 1 Node Reference

| Node class_type            | ComfyUI custom node package |
| -------------------------- | --------------------------- |
| `VHS_LoadVideo`            | ComfyUI-VideoHelperSuite    |
| `DWPose_Estimator`         | comfyui-controlnet-aux      |
| `DensePose`                | comfyui-controlnet-aux      |
| `ControlNet_LineArt_Anime` | comfyui-controlnet-aux      |
| `Canny_Edge`               | comfyui-controlnet-aux      |
| `ZoeDepth`                 | comfyui-controlnet-aux      |
| `Unimatch_Optical_Flow`    | ComfyUI-UniMatch            |

## Stage 2 Node Reference

| Node class_type              | Role                                       |
| ---------------------------- | ------------------------------------------ |
| `Layered_Model_Unload`       | Purge Stage 1 preprocessors from VRAM      |
| `GGUF_Loader`                | Load Wan 2.6 NVFP4 model (~4.2GB)          |
| `CLIPTextEncode`             | Positive / negative text conditioning      |
| `LoadImage`                  | Face reference image for identity anchoring |
| `IP_Adapter_FaceID_Plus`     | Identity injection                         |
| `Reference_Only`             | Secondary identity anchor                  |
| `Style_Transfer_Block`       | Anime latent alignment (CFG 1.5)           |
| `EmptyLatentVideo`           | Noise latent source (empty mode)           |
| `VAEEncode`                  | Encoded video latent source (vae_encode mode) |
| `Flow_Guided_Noise_Injection`| Skew noise along Unimatch motion vectors   |
| `FreeLong`                   | Spectral blending for cross-chunk stability |
| `KSampler`                   | Main diffusion sampling (tiled)            |

## Plans Location

Implementation plans live in `docs/superpowers/plans/`. Each plan covers exactly one stage and produces independently testable software.
