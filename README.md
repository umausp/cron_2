# cron_2 — Agyata model lab

A dedicated, isolated ground to **build and prove heavy / unverified AI models** on GitHub Actions
runners (which have `onnxruntime`, unlike the dev laptop) **before** they land in the live
`content-sync-worker` pipeline. Nothing here touches the 34 live crons.

Why a separate repo (not "dedicated hardware"): GitHub Actions already gives every job its own isolated
ephemeral VM (4 vCPU), so crons don't share a CPU. What this repo buys is a **safe build/A-B ground** + a
**separate concurrency budget** — an unproven model can't take a channel dark from here.

## Current probe: Audio8-TTS-Preview-0.6B-ONNX-INT4

- **License:** Apache-2.0 (commercial-safe — unlike XTTS-v2's non-commercial CPML, which we dropped).
- **Why:** the commercial-safe, CPU-ONNX, multilingual (de/nl/pl/es/it/fr + more) candidate to replace
  **Piper** as the primary offline voice on the 9 native channels (Kokoro can't speak those; edge throttles).
- **Reality check:** Audio8 is **model-files-only** + a separate GitHub runtime (`Audio8-AI/Audio8_TTS`,
  `setup.sh` + a server) and is **voice-cloning** (register a reference voice, then synth). So it is NOT a
  drop-in ONNX worker like our depth/RVM/ESRGAN — this probe does the full round-trip to see if it's viable.

### Run it
Actions → **audio8-prove** → Run workflow. It clones the runtime, downloads the model, sets up, registers
a reference voice, synthesizes test lines in several languages, and uploads the WAVs as an artifact
(`audio8-samples`) to **listen to and judge**. Green + good-sounding samples = go; then the engine code
lands in the pipeline as a flagged ONNX engine (like ESRGAN), and the natives repoint off Piper.

See the main-repo memory `project_model_lab_repo` for the full decision trail.
