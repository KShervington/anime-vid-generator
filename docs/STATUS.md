# Project Kinetic — Implementation Status

## Stages

| Stage | Description                     | Status         | Tests   |
| ----- | ------------------------------- | -------------- | ------- |
| 1     | Multi-Modal Motion Extraction   | ✅ Complete    | 58/58   |
| 2     | Temporal Latent Engine          | ✅ Complete    | 109/109 |
| 3     | Dynamic VFX & Masked Inpainting | ⬜ Not started | —       |
| 4     | High-Resolution Synthesis       | ⬜ Not started | —       |

---

## Stage 1 — Multi-Modal Motion Extraction ✅

**Plan:** `docs/superpowers/plans/2026-04-09-stage1-motion-extraction.md`

### Deliverables

| Component                                             | File                                                       | Status |
| ----------------------------------------------------- | ---------------------------------------------------------- | ------ |
| Config (HardwareConfig, Stage1Config, PipelineConfig) | `src/anime_vid_generator/config.py`                        | ✅     |
| ComfyUI HTTP + WebSocket client                       | `src/anime_vid_generator/client.py`                        | ✅     |
| Typer CLI (`kinetic stage1`)                          | `src/anime_vid_generator/cli.py`                           | ✅     |
| WorkflowNode + node factories                         | `src/anime_vid_generator/workflow/nodes.py`                | ✅     |
| WorkflowBuilder                                       | `src/anime_vid_generator/workflow/builder.py`              | ✅     |
| Organic bus (DWPose + DensePose)                      | `src/anime_vid_generator/workflow/stages/stage1_motion.py` | ✅     |
| Rigid bus (LineArt + Canny + ZoeDepth)                | `src/anime_vid_generator/workflow/stages/stage1_motion.py` | ✅     |
| Temporal unification (Unimatch Optical Flow)          | `src/anime_vid_generator/workflow/stages/stage1_motion.py` | ✅     |

### Commits

| Hash      | Description                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------- |
| `4ac0d55` | chore: set up src/ layout, package markers, and dev dependencies                                  |
| `30fb83d` | feat: add pydantic config for hardware and Stage 1 pipeline settings                              |
| `82d28ba` | feat: add WorkflowNode dataclass and Stage 1 ComfyUI node factories                               |
| `0b6b9d3` | feat: add WorkflowBuilder with auto-ID assignment and ComfyUI JSON serialization                  |
| `c0582a0` | feat: add Stage 1 motion extraction — organic bus, rigid bus, temporal unification, and assembler |
| `29a37d4` | feat: add ComfyUI HTTP + WebSocket client                                                         |
| `6faf5c0` | feat: add Typer CLI with stage1 command and --dry-run workflow preview                            |
| `7e7fcec` | fix: URL scheme replacement, add monitor_progress test, assert detect_body                        |
| `b956750` | chore: add project documentation — README, CLAUDE.md, TDD, and Python config files                |

---

## Stage 2 — Temporal Latent Engine ✅

**Plan:** `docs/superpowers/plans/2026-04-10-stage2-temporal-latent-engine.md`

### Deliverables

| Component                                                              | File                                                          | Status |
| ---------------------------------------------------------------------- | ------------------------------------------------------------- | ------ |
| Stage2Config (model, identity, sampler, latent settings)               | `src/anime_vid_generator/config.py`                           | ✅     |
| 12 Stage 2 node factory functions                                      | `src/anime_vid_generator/workflow/nodes.py`                   | ✅     |
| Model loading bus (Layered_Model_Unload + GGUF_Loader)                 | `src/anime_vid_generator/workflow/stages/stage2_generation.py`| ✅     |
| Conditioning bus (CLIPTextEncode ×2)                                   | `src/anime_vid_generator/workflow/stages/stage2_generation.py`| ✅     |
| Identity bus (IP-Adapter FaceID Plus + Reference Only + Style Transfer) | `src/anime_vid_generator/workflow/stages/stage2_generation.py`| ✅     |
| Latent bus (EmptyLatentVideo or VAEEncode + Flow_Guided_Noise_Injection) | `src/anime_vid_generator/workflow/stages/stage2_generation.py`| ✅     |
| Generation bus (FreeLong + KSampler with tiled sampling)               | `src/anime_vid_generator/workflow/stages/stage2_generation.py`| ✅     |
| `build_stage2_workflow` assembler (Stage 1 + Stage 2 combined)         | `src/anime_vid_generator/workflow/stages/stage2_generation.py`| ✅     |
| Typer CLI (`kinetic stage2`)                                           | `src/anime_vid_generator/cli.py`                              | ✅     |

### Commits

| Hash      | Description                                                                                     |
| --------- | ----------------------------------------------------------------------------------------------- |
| `2b95122` | feat: add Stage2Config with sampler, identity, and latent settings                              |
| `1103cc6` | feat: add 12 Stage 2 ComfyUI node factory functions                                             |
| `b9e4b76` | feat: add model loading bus — Layered_Model_Unload + GGUF_Loader                                |
| `0a8b01b` | feat: add conditioning bus — positive and negative CLIPTextEncode                               |
| `44cca3d` | feat: add identity bus — IP-Adapter FaceID Plus, Reference Only, Style Transfer Block          |
| `078f0f0` | feat: add latent bus — EmptyLatentVideo/VAEEncode + Flow_Guided_Noise_Injection                 |
| `a12569a` | feat: add generation bus — FreeLong spectral blending and KSampler                             |
| `eb2748d` | feat: add build_stage2_workflow assembler — chains Stage 1 preprocessing with Stage 2 generation |
| `fd5dfcc` | feat: add kinetic stage2 CLI command with --dry-run, --prompt, --vae-encode, --denoise          |
| `533613e` | fix: generalize _submit_and_monitor progress messages for all stages                            |
| `9bddfc3` | fix: add KSampler seed field, remove unused fixtures, align assembler node order                |

---

---

## Stage 3 — Dynamic VFX & Masked Inpainting ⬜

**Scope (from TDD):**

- SAM 3 (Segment Anything Video) for effect emitter tracking (sword tip, tire contact patch)
- Mask_Dilate node driven by Optical Flow vectors
- Secondary VAE Encode loop masked to emitter trajectory
- VFX LoRA injection (e.g., Ufotable_Fire_Trails) at high CFG (8.0+)

---

## Stage 4 — High-Resolution Synthesis ⬜

**Scope (from TDD):**

- Tiled_VAE_Decode with 512×512 spatial tiles, 8-frame temporal chunks
- RTX_VSR_Upscaler node for driver-level 4K upscaling via Tensor Cores
- Flow-Guided_Noise_Injection for high-velocity subject sharpness
- Prompt_Schedule node for impact frame handling (ControlNet drop 1.0→0.3, CFG spike)
