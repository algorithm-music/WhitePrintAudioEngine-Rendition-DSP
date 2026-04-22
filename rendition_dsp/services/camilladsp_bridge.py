# -*- coding: utf-8 -*-
"""
CamillaDSP Bridge — generates YAML configs and invokes the CamillaDSP binary
for high-performance EQ, compression, limiting, and dithering.

This replaces the scipy-based implementations of these stages with
CamillaDSP's Rust engine (64-bit float, SIMD-optimized).
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger("rendition_dsp.camilladsp_bridge")

# Path to the CamillaDSP binary — built during Docker image construction.
_CAMILLADSP_BIN = os.environ.get(
    "CAMILLADSP_BIN",
    shutil.which("camilladsp") or "/usr/local/bin/camilladsp",
)

# Temporary directory for intermediate files.
_TMP_DIR = os.environ.get("TMPDIR", "/tmp")


def _safe_float(val: Any, default: float) -> float:
    """Extract a float from a param dict value, with fallback."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def build_config(
    input_path: str,
    output_path: str,
    sr: int,
    params: dict,
    gain_adj: float = 0.0,
) -> dict:
    """Build a CamillaDSP YAML config dict from adopted_params.

    Maps the AI-generated mastering parameters to CamillaDSP's native
    filter/processor types.
    """
    # ── EQ parameters ──────────────────────────────────────────────
    eq_bands = params.get("eq_bands", [])
    # Fallback: construct from flat params if eq_bands list is absent
    if not eq_bands:
        eq_bands = [
            {
                "type": "Lowshelf",
                "freq": _safe_float(params.get("eq_low_freq"), 80.0),
                "gain": _safe_float(params.get("eq_low_gain"), 0.0),
                "q": _safe_float(params.get("eq_low_q"), 0.7),
            },
            {
                "type": "Peaking",
                "freq": _safe_float(params.get("eq_low_mid_freq"), 250.0),
                "gain": _safe_float(params.get("eq_low_mid_gain"), 0.0),
                "q": _safe_float(params.get("eq_low_mid_q"), 1.0),
            },
            {
                "type": "Peaking",
                "freq": _safe_float(params.get("eq_mid_freq"), 1000.0),
                "gain": _safe_float(params.get("eq_mid_gain"), 0.0),
                "q": _safe_float(params.get("eq_mid_q"), 1.0),
            },
            {
                "type": "Peaking",
                "freq": _safe_float(params.get("eq_hi_mid_freq"), 4000.0),
                "gain": _safe_float(params.get("eq_hi_mid_gain"), 0.0),
                "q": _safe_float(params.get("eq_hi_mid_q"), 1.0),
            },
            {
                "type": "Highshelf",
                "freq": _safe_float(params.get("eq_high_freq"), 8000.0),
                "gain": _safe_float(params.get("eq_high_gain"), 0.0),
                "q": _safe_float(params.get("eq_high_q"), 0.7),
            },
        ]

    # ── Compressor parameters ──────────────────────────────────────
    comp_attack = _safe_float(params.get("comp_attack"), 0.01)
    comp_release = _safe_float(params.get("comp_release"), 0.15)
    comp_threshold = _safe_float(params.get("comp_threshold"), 0.0)
    comp_ratio = _safe_float(params.get("comp_ratio"), 1.0)
    comp_makeup = _safe_float(params.get("comp_makeup_gain"), 0.0)

    # ── Limiter parameters ─────────────────────────────────────────
    limiter_ceil_db = _safe_float(params.get("limiter_ceil_db"), 0.0)

    # ── Dither parameters ──────────────────────────────────────────
    dither_bits = int(_safe_float(params.get("dither_bits"), 24))
    dither_enabled = params.get("dither_enabled", True)

    # ── Input gain (convergence adjustment) ────────────────────────
    input_gain_db = _safe_float(params.get("input_gain_db"), 0.0) + gain_adj

    # ── Build filters dict ─────────────────────────────────────────
    filters = {}
    filter_names = []

    # DC remove (5 Hz highpass)
    filters["dc_remove"] = {
        "type": "Biquad",
        "parameters": {
            "type": "HighpassFO",
            "freq": 5.0,
        },
    }
    filter_names.append("dc_remove")

    # Input gain
    if abs(input_gain_db) > 0.001:
        filters["input_gain"] = {
            "type": "Gain",
            "parameters": {
                "gain": round(input_gain_db, 4),
                "inverted": False,
                "mute": False,
            },
        }
        filter_names.append("input_gain")

    # Parametric EQ bands
    for i, band in enumerate(eq_bands):
        band_type = band.get("type", "Peaking")
        band_freq = _safe_float(band.get("freq"), 1000.0)
        band_gain = _safe_float(band.get("gain"), 0.0)
        band_q = _safe_float(band.get("q"), 1.0)

        # Skip bands with negligible gain
        if abs(band_gain) < 0.05:
            continue

        # Clamp frequency to valid range
        nyquist = sr / 2.0
        band_freq = max(10.0, min(band_freq, nyquist - 100))

        name = f"eq_band_{i}"
        if band_type in ("Lowshelf", "LowShelf"):
            filters[name] = {
                "type": "Biquad",
                "parameters": {
                    "type": "Lowshelf",
                    "freq": round(band_freq, 1),
                    "gain": round(band_gain, 2),
                    "q": round(max(0.1, band_q), 3),
                },
            }
        elif band_type in ("Highshelf", "HighShelf"):
            filters[name] = {
                "type": "Biquad",
                "parameters": {
                    "type": "Highshelf",
                    "freq": round(band_freq, 1),
                    "gain": round(band_gain, 2),
                    "q": round(max(0.1, band_q), 3),
                },
            }
        else:  # Peaking
            filters[name] = {
                "type": "Biquad",
                "parameters": {
                    "type": "Peaking",
                    "freq": round(band_freq, 1),
                    "gain": round(band_gain, 2),
                    "q": round(max(0.1, band_q), 3),
                },
            }
        filter_names.append(name)

    # Limiter (soft clip)
    filters["limiter"] = {
        "type": "Limiter",
        "parameters": {
            "clip_limit": round(limiter_ceil_db, 2),
            "soft_clip": bool(params.get("soft_clip_enabled", False)),
        },
    }

    # Dither
    if dither_enabled:
        # CamillaDSP dither type depends on sample rate
        dither_type = "Lipshitz441" if sr <= 48000 else "None"
        filters["dither"] = {
            "type": "Dither",
            "parameters": {
                "type": dither_type,
                "bits": dither_bits,
            },
        }

    # ── Build processors dict ──────────────────────────────────────
    processors = {}
    processors["compressor"] = {
        "type": "Compressor",
        "parameters": {
            "channels": 2,
            "attack": round(max(0.0001, comp_attack), 6),
            "release": round(max(0.001, comp_release), 4),
            "threshold": round(comp_threshold, 1),
            "factor": round(max(1.0, comp_ratio), 2),
            "makeup_gain": round(comp_makeup, 2),
            "soft_clip": bool(params.get("soft_clip_enabled", False)),
            "clip_limit": round(limiter_ceil_db, 2),
        },
    }

    # ── Build pipeline ─────────────────────────────────────────────
    pipeline = []

    # Filters: DC remove + gain + EQ bands
    if filter_names:
        pipeline.append({
            "type": "Filter",
            "channels": [0, 1],
            "names": filter_names,
        })

    # Compressor
    pipeline.append({
        "type": "Processor",
        "name": "compressor",
    })

    # Post-processing filters: limiter + dither
    post_filter_names = ["limiter"]
    if dither_enabled:
        post_filter_names.append("dither")
    pipeline.append({
        "type": "Filter",
        "channels": [0, 1],
        "names": post_filter_names,
    })

    # ── Assemble full config ───────────────────────────────────────
    config = {
        "devices": {
            "samplerate": sr,
            "chunksize": 4096,
            "capture": {
                "type": "WavFile",
                "filename": input_path,
            },
            "playback": {
                "type": "File",
                "channels": 2,
                "format": "S32_LE",
                "wav_header": True,
                "filename": output_path,
            },
        },
        "filters": filters,
        "processors": processors,
        "pipeline": pipeline,
    }

    return config


def run_camilladsp(
    input_path: str,
    output_path: str,
    sr: int,
    params: dict,
    gain_adj: float = 0.0,
    timeout_sec: float = 120.0,
) -> Dict[str, Any]:
    """Generate a YAML config and run CamillaDSP to process the audio file.

    Returns:
        dict with keys: success, config_path, stdout, stderr, returncode
    """
    config = build_config(input_path, output_path, sr, params, gain_adj)

    # Write YAML config to a temp file
    config_fd, config_path = tempfile.mkstemp(
        suffix=".yml", prefix="cdsp_", dir=_TMP_DIR,
    )
    try:
        with os.fdopen(config_fd, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(
            f"CamillaDSP config written to {config_path}, "
            f"input={input_path}, output={output_path}, sr={sr}"
        )

        # Validate config first
        check_result = subprocess.run(
            [_CAMILLADSP_BIN, "--check", config_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if check_result.returncode != 0:
            logger.error(
                f"CamillaDSP config validation failed: {check_result.stderr}"
            )
            return {
                "success": False,
                "error": f"Config validation failed: {check_result.stderr}",
                "config_path": config_path,
                "returncode": check_result.returncode,
            }

        # Run processing
        result = subprocess.run(
            [_CAMILLADSP_BIN, config_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        success = result.returncode == 0 and os.path.exists(output_path)
        if not success:
            logger.error(
                f"CamillaDSP processing failed (rc={result.returncode}): "
                f"{result.stderr}"
            )
        else:
            logger.info("CamillaDSP processing completed successfully.")

        return {
            "success": success,
            "config_path": config_path,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        logger.error(f"CamillaDSP timed out after {timeout_sec}s")
        return {
            "success": False,
            "error": f"CamillaDSP timed out after {timeout_sec}s",
            "config_path": config_path,
            "returncode": -1,
        }
    except FileNotFoundError:
        logger.error(f"CamillaDSP binary not found at {_CAMILLADSP_BIN}")
        return {
            "success": False,
            "error": f"CamillaDSP binary not found at {_CAMILLADSP_BIN}",
            "config_path": config_path,
            "returncode": -1,
        }
    finally:
        # Clean up config file (output WAV is kept)
        try:
            if os.path.exists(config_path):
                os.remove(config_path)
        except OSError:
            pass
