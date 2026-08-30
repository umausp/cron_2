#!/usr/bin/env python3
# cron_2 MODEL-LAB probe (user 2026-08-30): prove Chatterbox Multilingual (MIT license, 23 languages incl.
# Hindi) can synthesise clean speech on a CPU-ONLY GitHub runner at an acceptable REAL-TIME FACTOR, BEFORE
# it lands in the content-sync-worker pipeline as a flagged TTS engine to replace the robotic / throttled
# edge voices on the 9 native channels (esp. the still-robotic hi-IN-SwaraNeural) — Phase 2 of the
# voice-naturalness initiative. All 9 native edge langs are covered: hi/es/ja/nl/sv/no/da/fi/pl.
#
# The human listens to the per-language WAVs and reads the RTF in metrics.json to make the go/no-go call.
# Best-effort + logs loudly; failure IS the signal.
#
# GO/NO-GO on RTF (synth wall-seconds / audio seconds, on ~4 vCPU):
#   RTF < 1  = faster than realtime (excellent) | RTF 1-5 = usable for the overnight batch factory
#   RTF > ~8 = likely too slow even for batch (a 10-min longform > 80 min of TTS alone) -> no-go on CPU.
#
# GENERATION KNOBS (user 2026-08-30 "use the right values; can it be dynamic per beat/pitch/delivery?"):
# Chatterbox exposes exaggeration / cfg_weight / temperature; the seed is set globally before generate.
#   • exaggeration (0.5 default): emotional intensity. HIGHER = more dramatic AND faster (~0.7 expressive).
#   • cfg_weight   (0.5 default): guidance/PACE. LOWER (~0.3) = slower, more deliberate delivery.
#   • temperature  (0.8 default): sampling variance. Lower = steadier/consistent; higher = more dynamic.
#   • seed:        fixed per render for a CONSISTENT anchor timbre across beats + reproducible A/B.
# These are the SAME delivery levers the longform prosody planner (prosody.mjs) already computes per beat
# (beat-arc rate -> cfg_weight; arousal/emotion -> exaggeration/temperature). PITCH + sub-beat EMPHASIS are
# NOT generate knobs here — Chatterbox would be an OFFLINE-class engine, so the existing offline DSP pass
# (tts_dsp.mjs, which is a no-op on edge) applies the pitch shift + slow/lift-the-fact window to its wav,
# exactly as it already does for kokoro/piper. So a landed Chatterbox engine gets BOTH beat-level delivery
# AND the precise sub-beat emphasis landing edge cannot. This probe DEMONSTRATES the beat dynamism now
# (CBX_DELIVERY_DEMO renders one line under hook/stakes/meaning presets) so it can be heard before wiring.
import os
import sys
import time
import json
import random
import traceback

DEVICE = os.environ.get("CBX_DEVICE", "cpu")  # pin CPU on the runner; "mps"/"cuda" for local checks

# lang_id -> a realistic NEWS sentence. Weakest edge langs FIRST (hi = the priority). Each maps 1:1 to a
# native edge channel: hi=bharat, es=es, ja=jp, nl=nl, sv=sv, no=no, da=da, fi=fi, pl=pl. en = a control.
DEFAULT_LINES = {
    "hi": "नमस्ते, ये आज की मुख्य ख़बरें हैं। दुनिया भर से ताज़ा समाचार, सिर्फ़ आपके लिए।",
    "es": "Buenas noches, estas son las noticias principales de hoy desde todo el mundo.",
    "ja": "こんばんは。今日の主要なニュースを、世界各地からお伝えします。",
    "nl": "Goedenavond, dit is het belangrijkste nieuws van vandaag uit de hele wereld.",
    "en": "Good evening. Here are today's top stories from around the world.",
}

# DELIVERY PRESETS = beat-role -> (exaggeration, cfg_weight, temperature), mapped from the prosody.mjs
# PROFILES so the mapping is provably the same one production would use. "news" is the steady anchor
# baseline every language renders at; hook/stakes/meaning are the beat-role variants the demo renders to
# PROVE per-beat modulation is audible (a quick hook vs a grave, deliberate stakes vs a warm meaning).
PRESETS = {
    "neutral": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},   # model default (control)
    "news":    {"exaggeration": 0.5, "cfg_weight": 0.4, "temperature": 0.7},   # steady anchor baseline
    "hook":    {"exaggeration": 0.6, "cfg_weight": 0.55, "temperature": 0.85}, # brighter + quicker
    "stakes":  {"exaggeration": 0.45, "cfg_weight": 0.3, "temperature": 0.7},  # grave, deliberate
    "meaning": {"exaggeration": 0.55, "cfg_weight": 0.3, "temperature": 0.7},  # warm, slow landing
}


def parse_lines():
    """CBX_LINES = 'hi:...|es:...' pipe-separated lang:text pairs; empty -> DEFAULT_LINES."""
    env = os.environ.get("CBX_LINES", "").strip()
    if not env:
        return DEFAULT_LINES
    out = {}
    for pair in env.split("|"):
        pair = pair.strip()
        if ":" in pair:
            k, v = pair.split(":", 1)
            if k.strip() and v.strip():
                out[k.strip()] = v.strip()
    return out or DEFAULT_LINES


def base_params():
    """Start from a preset (CBX_PRESET, default 'news'), then let CBX_EXAG/CBX_CFG/CBX_TEMP override."""
    preset = os.environ.get("CBX_PRESET", "news")
    p = dict(PRESETS.get(preset, PRESETS["news"]))

    def num(env, key):
        v = os.environ.get(env, "").strip()
        if v:
            try:
                p[key] = float(v)
            except ValueError:
                pass
    num("CBX_EXAG", "exaggeration")
    num("CBX_CFG", "cfg_weight")
    num("CBX_TEMP", "temperature")
    return preset, p


def seed_all(seed, torch):
    """Fixed seed before EACH generate -> reproducible A/B + a consistent voice across beats."""
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass


def synth(model, ta, torch, text, lang, params, seed, tag):
    """One generation + measurement. Returns a result dict; writes outputs/<tag>.wav."""
    seed_all(seed, torch)
    t0 = time.time()
    wav = model.generate(
        text, language_id=lang,
        exaggeration=params["exaggeration"],
        cfg_weight=params["cfg_weight"],
        temperature=params["temperature"],
    )
    synth_secs = time.time() - t0
    if hasattr(wav, "dim") and wav.dim() == 1:
        wav = wav.unsqueeze(0)
    path = f"outputs/{tag}.wav"
    sr = int(getattr(model, "sr", 24000))
    ta.save(path, wav, sr)
    n = int(wav.shape[-1])
    audio_secs = n / sr if sr else 0.0
    rtf = (synth_secs / audio_secs) if audio_secs else None
    print(f"[{tag}] synth={synth_secs:.1f}s audio={audio_secs:.1f}s RTF={rtf if rtf is None else round(rtf,2)}"
          f" exag={params['exaggeration']} cfg={params['cfg_weight']} temp={params['temperature']}"
          f" chars={len(text)} -> {path}", flush=True)
    return {
        "tag": tag, "lang": lang, "synth_secs": round(synth_secs, 2),
        "audio_secs": round(audio_secs, 2), "rtf": round(rtf, 2) if rtf is not None else None,
        "chars": len(text), "params": params, "ok": True,
    }


def main():
    lines = parse_lines()
    t3 = os.environ.get("CBX_T3", "v3") or "v3"
    seed = int(os.environ.get("CBX_SEED", "0") or "0")
    preset_name, params = base_params()
    demo_lang = os.environ.get("CBX_DELIVERY_DEMO", "").strip()  # e.g. "hi" -> render it under all presets

    import torch
    import torchaudio as ta
    # Chatterbox applies an inaudible Perth watermark by default. Its implicit-watermarker impl is an
    # OPTIONAL sub-dependency that isn't always importable (perth ships a no-op DummyWatermarker fallback
    # for exactly this). Watermarking is irrelevant to a quality/RTF probe, so fall back to the dummy ONLY
    # when the real one didn't import — this keeps the probe running instead of dying in the constructor.
    import perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
        print("(perth implicit watermarker unavailable -> DummyWatermarker no-op; N/A to quality/RTF)", flush=True)
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    print(f"torch={torch.__version__} device={DEVICE} t3_model={t3} preset={preset_name} params={params} "
          f"seed={seed} threads={torch.get_num_threads()} cuda={torch.cuda.is_available()}", flush=True)

    t_load = time.time()
    # §10 version-tolerance: newer chatterbox-tts selects the checkpoint via t3_model="v3"/"v2"; 0.1.7
    # (our pinned, locally-verified build) has a single bundled multilingual checkpoint and no such arg.
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE, t3_model=t3)
    except TypeError:
        print("(from_pretrained has no t3_model on this build — loading the bundled checkpoint)", flush=True)
        model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
    load_secs = time.time() - t_load
    sr = int(getattr(model, "sr", 24000))
    print(f"model loaded in {load_secs:.1f}s, sr={sr}", flush=True)

    os.makedirs("outputs", exist_ok=True)
    results = []
    # 1) every language at the steady "news" baseline (this is the production-intended default read).
    for lang, text in lines.items():
        try:
            results.append(synth(model, ta, torch, text, lang, params, seed, lang))
        except Exception as e:  # noqa: BLE001 — one bad lang must not kill the rest
            print(f"[{lang}] FAILED: {e}", flush=True)
            traceback.print_exc()
            results.append({"tag": lang, "lang": lang, "ok": False, "error": str(e)})

    # 2) DELIVERY DEMO: render ONE line under the beat-role presets so per-beat dynamism is AUDIBLE.
    if demo_lang and demo_lang in lines:
        text = lines[demo_lang]
        for role in ("hook", "stakes", "meaning", "news"):
            try:
                results.append(synth(model, ta, torch, text, demo_lang, PRESETS[role], seed, f"{demo_lang}__{role}"))
            except Exception as e:  # noqa: BLE001
                print(f"[{demo_lang}__{role}] FAILED: {e}", flush=True)
                traceback.print_exc()

    summary = {
        "device": DEVICE, "t3_model": t3, "sr": sr, "seed": seed,
        "base_preset": preset_name, "base_params": params,
        "load_secs": round(load_secs, 2), "torch": torch.__version__,
        "threads": torch.get_num_threads(), "results": results,
    }
    with open("outputs/metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("=== METRICS ===", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    if not any(r.get("ok") for r in results):
        print("::error::no language synthesized — Chatterbox Multilingual did not run on CPU", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
