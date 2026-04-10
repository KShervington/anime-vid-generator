# **TECHNICAL DESIGN DOCUMENT (TDD): PROJECT KINETIC**

**Author:** Lead Technical Director / Senior AI Workflow Architect

**Target Architecture:** ComfyUI (Build 2026.x)

**Target Hardware:** NVIDIA RTX 4070 Ti Super (16GB VRAM, Ada Lovelace)

**Output Specs:** 10–30s, 24fps (240–720 frames), 4K Output (via VSR)

**Aesthetic Profile:** Cinematic Anime (Ufotable/MAPPA), Thick Line-Art, Heavy VFX

## **Executive Summary**

This document outlines the node-logic architecture for a state-of-the-art Video-to-Video (Vid2Vid) pipeline capable of matching Kling 3.0 and Seedance 2.0 temporal coherency. Running a 30-second continuous sequence on a 16GB VRAM envelope requires extreme latent management. We achieve this via multi-modal motion bifurcation, NVFP4 model quantization, and temporal latent chunking.

## **Stage 1: Multi-Modal Motion Extraction (Organic vs. Rigid)**

To handle the diametric tracking requirements of human martial arts and rigid-body vehicle dynamics, the pre-processing stack must be bifurcated into two distinct ControlNet buses.

- **Organic Bus (Human Choreography):**
  - We utilize a combined DWPose_Estimator (for dense skeletal, hand, and facial landmarking) and a DensePose node for volumetric mapping.
  - _Why:_ Traditional OpenPose lacks z-depth estimation. DensePose provides the base model (Wan 2.6) with occlusion data, crucial when a character crosses their arms or spins.
- **Rigid Bus (Vehicle Dynamics):**
  - We route the performance car source footage through a ControlNet_LineArt_Anime and Canny_Edge preprocessor stack, bound by a ZoeDepth mask.
  - _Why:_ Skeletal tracking fails on mechanical hard-surfaces. Canny/LineArt forces the AI to respect the rigid geometry of the car during 180° drifts, preventing the chassis from "morphing" or "squashing" like organic tissue.
- **Temporal Unification:**
  - Both buses are normalized using an Unimatch_Optical_Flow node. This extracts the motion vectors (pixel displacement between frames **t** and **t+1**) and encodes them into a flow map, ensuring the structural guidelines remain temporally locked before entering the latent space.

## **Stage 2: Temporal Latent Engine & Identity Anchoring**

Running 720 frames of generation on a 16GB card requires migrating away from FP16 weights to NVIDIA's Ada/Blackwell optimized formats.

- **The Engine:** We deploy the **Wan 2.6 (or SVI Pro)** base model quantized to **NVFP4** format, loaded via a GGUF_Loader. This reduces the model footprint from \~14GB to \~4.2GB, preserving massive VRAM headroom for the context window.
- **Identity Lock (Zero-Drift Strategy):**
  - Over 30 seconds, traditional LoRAs experience "identity decay" (costumes simplify, eye shapes drift).
  - We implement an IP-Adapter FaceID Plus node alongside a Reference-Only injection.
  - _Crucial Node Logic:_ The IP-Adapter output is routed through a Style_Transfer_Block node set to a CFG of 1.5. This prevents the photorealistic tendencies of IP-Adapter from overriding the thick-lined MAPPA/Ufotable character LoRA, forcing the identity features into the anime latent space.
- **Temporal Stability (Latent Chaining):**
  - We employ a "Context Overlap" strategy using a 32-frame context window with an 8-frame overlap (e.g., Frames 1-32, then 24-56).
  - To prevent background morphing across chunks, we route the generation through a FreeLong (Spectral Blending) node. FreeLong isolates the low-frequency spatial components (backgrounds, ambient lighting) in the Fourier domain and forces alignment across the latent chunks.

## **Stage 3: Dynamic VFX & Masked Inpainting**

The Ufotable aesthetic relies heavily on volumetric 2D effects (fire, water, lightning) composited accurately in 3D space. Instead of prompting these globally, we use a targeted inpainting loop.

- **Motion Tracking the Effect Emitters:**
  - We deploy SAM 3 (Segment Anything Video) to isolate the tip of the sword or the rear tires of the drifting car.
  - The tracked mask is expanded via a Mask_Dilate node and driven by the previously generated Optical Flow vectors.
- **Elemental Inpainting Sub-Routine:**
  - We create a secondary VAE Encode loop that masks out everything except the dilated emitter trajectory.
  - A highly stylized VFX LoRA (e.g., "Ufotable_Fire_Trails") is injected here with a high CFG scale (8.0+).
  - _Result:_ Because the noise injection is bounded by the mask and guided by optical flow, the generated fire trails physically wrap around the 3D trajectory of the sword swing or drift smoke, rather than floating detached in the foreground.

## **Stage 4: High-Resolution Synthesis & Finishing**

Decoding 1080p latents for hundreds of frames is the most common point of Out-Of-Memory (OOM) failure.

- **Tiled VAE Decoding:** We utilize Tiled_VAE_Decode with spatial tiling (512x512 tiles) and temporal chunking (decoding 8 frames at a time).
- **Hardware Upscaling (RTX VSR):**
  - Instead of using traditional ESRGAN nodes (which consume standard VRAM), we route the decoded 1080p RGB tensor directly into an RTX_VSR_Upscaler node.
  - _Why:_ This node communicates directly with the RTX 4070 Ti Super's Tensor Cores at the driver level, offloading the 4K upscaling and artifact reduction entirely from the standard PyTorch memory pool, freeing up the pipeline.

## **Hardware Optimization (The 16GB VRAM Survival Guide)**

To prevent CUDA OOM errors during this heavy pipeline, strict memory hooks must be established in ComfyUI:

1. **VRAM Partitioning:**
   - Base Model (NVFP4): \~4.2GB
   - ControlNet Stack (LineArt \+ DWPose loaded dynamically): \~2.0GB
   - Context Window (32-frame latent chunks): \~5.5GB
   - Buffer / PyTorch Overhead: \~4.3GB
2. **Memory Hooks:** Enable \--lowvram argument execution specifically for the ControlNet bus. Use the Layered_Model_Unload node after Stage 1 to aggressively purge the preprocessor models from VRAM before initializing the Wan 2.6 generation block.
3. **Tiled Sampling:** Use Temporal_Tiled_Sampling in the K-Sampler. Do not attempt to process the entire 720-frame batch simultaneously.

## **Troubleshooting High-Speed Choreography**

High-velocity subjects (drifts, rapid sword strikes) inherently cause the base model to "smear" or produce multi-limbed ghosting artifacts.

- **Combating Motion Blur (Flow-Guided Sampling):**
  - We implement Flow-Guided_Noise_Injection. Instead of injecting uniform Gaussian noise during the diffusion process, we skew the noise distribution along the Unimatch motion vectors. This pre-aligns the latents with the subject's trajectory, allowing the AI to render sharp character details even during a 180° camera pan.
- **Handling "Impact Frames":**
  - Anime action relies on "Impact Frames" (1-2 frames of high-contrast, abstract, ink-splash art at the apex of a strike).
  - _Technical Implementation:_ We use a Prompt_Schedule node tied to the optical flow velocity. When the pixel velocity spikes above a defined threshold (the impact), the pipeline automatically drops the ControlNet weights from 1.0 to 0.3 for exactly two frames and spikes the CFG.
  - _Result:_ This temporarily releases the AI from strict anatomical adherence, allowing it to generate highly stylized, abstract smear frames, before immediately snapping back to structural lock in the subsequent frame.
