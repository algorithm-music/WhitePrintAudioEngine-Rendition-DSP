#!/usr/bin/env python3
"""
Local batch mastering — Full Pipeline via Concertmaster API.

Flow per track:
  POST /api/v1/jobs/master (input_path=local file, route=full)
  → Audition analyzes → Deliberation decides params → Rendition DSP masters

No hardcoded DSP params. Everything is decided by the AI pipeline.

Usage:
    1. Run start_local_pipeline.bat first (starts all 4 services)
    2. python batch_master_local.py [source_dir] [output_dir]

Defaults:
    source_dir = ~/Music
    output_dir = ~/Music/mastered_v3
"""

import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_master")

CONCERTMASTER_URL = os.environ.get("CONCERTMASTER_URL", "http://localhost:8080")
API_KEY = os.environ.get("CONCERTMASTER_API_KEY", "local-dev-key")
POLL_INTERVAL = 5  # seconds between status polls
POLL_TIMEOUT = 1800  # 30 minutes max per track

# Skip patterns — files that shouldn't be mastered
import re
import unicodedata

SKIP_PATTERNS = [
    "_mastered", "_remaster", "_spotify", "_beatport",
    "_ref_16bit", "_ref_24bit", "Maximized",
    "master_master", "master02", "master_aki", "master_dnbz",
    "master.wav",
]


def _sanitize_for_distribution(stem: str) -> str:
    """Sanitize filename for music distribution submission.

    Rules:
    - Strip leading/trailing whitespace
    - Remove characters illegal in most distribution platforms: # $ ( ) [ ] { } ! @ % ^ & * + = | \\ / < > ? " '
    - Collapse multiple spaces/underscores to single underscore
    - Strip leading track number prefixes like '001_', '#001 ', etc. and re-prepend as clean 3-digit
    - Remove '_mastered', '_remaster' suffixes if accidentally nested
    - Ensure no leading/trailing underscores
    """
    s = stem.strip()

    # Extract leading track number if present (e.g., "001_", "#001 ", "003_")
    track_num_match = re.match(r'^[#]?(\d{1,3})[_\s\-\.]+', s)
    track_num = None
    if track_num_match:
        track_num = int(track_num_match.group(1))
        s = s[track_num_match.end():]

    # Remove problematic characters for distribution platforms
    s = re.sub(r'[#$\(\)\[\]{}\!@%\^&\*\+=\|\\/<>\?"\'`~;:]+', '', s)

    # Replace spaces and multiple underscores with single underscore
    s = re.sub(r'[\s_]+', '_', s)

    # Remove stale suffixes
    s = re.sub(r'_?(mastered|remaster|spotify_master|beatport_master)$', '', s, flags=re.IGNORECASE)

    # Strip leading/trailing underscores and hyphens
    s = s.strip('_-')

    # Re-prepend track number if found
    if track_num is not None:
        s = f"{track_num:03d}_{s}"

    # Final safety: if empty after cleanup, use 'Untitled'
    if not s:
        s = "Untitled"

    return s


def should_skip(filename: str) -> bool:
    lower = filename.lower()
    for pat in SKIP_PATTERNS:
        if pat.lower() in lower:
            return True
    if any(f"({i})" in filename for i in range(1, 8)):
        return True
    if "(VIP)" in filename or "(Club Mix)" in filename:
        return True
    return False


def submit_job(client: httpx.Client, input_path: str, output_path: str) -> str:
    """Submit a mastering job to Concertmaster. Returns job_id."""
    resp = client.post(
        f"{CONCERTMASTER_URL}/api/v1/jobs/master",
        json={
            "input_path": input_path,
            "output_path": output_path,
            "route": "full",
        },
        headers={
            "X-Api-Key": API_KEY,
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["job_id"]


def poll_job(client: httpx.Client, job_id: str) -> dict:
    """Poll until job completes or fails. Returns result dict."""
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        resp = client.get(
            f"{CONCERTMASTER_URL}/api/v1/jobs/{job_id}",
            headers={"X-Api-Key": API_KEY},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "unknown")

        if status == "completed":
            return data
        elif status == "failed":
            raise RuntimeError(data.get("error", "Unknown pipeline error"))

        # Still processing — log stage
        stage = data.get("stage", "unknown")
        logger.info(f"    ... stage: {stage}")
        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Job {job_id} did not complete within {POLL_TIMEOUT}s")


def main():
    source_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Music")
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(source_dir, "mastered_v3")

    os.makedirs(output_dir, exist_ok=True)

    # Check Concertmaster is reachable
    try:
        r = httpx.get(f"{CONCERTMASTER_URL}/health", timeout=5.0)
        r.raise_for_status()
        logger.info(f"Concertmaster OK: {r.json()}")
    except Exception as e:
        logger.error(f"Cannot reach Concertmaster at {CONCERTMASTER_URL}: {e}")
        logger.error("Run start_local_pipeline.bat first!")
        sys.exit(1)

    # Collect tracks
    all_wavs = sorted(glob.glob(os.path.join(source_dir, "*.wav")))
    tracks = []
    for path in all_wavs:
        fname = os.path.basename(path)
        if should_skip(fname):
            continue
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb < 5:
            continue
        tracks.append(path)

    logger.info(f"═══════════════════════════════════════════════")
    logger.info(f"  Full Pipeline Batch: {len(tracks)} tracks")
    logger.info(f"  Source: {source_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Pipeline: Audition → Deliberation → Rendition DSP")
    logger.info(f"═══════════════════════════════════════════════")

    results = []
    failed = []

    with httpx.Client() as client:
        for i, path in enumerate(tracks, 1):
            fname = os.path.basename(path)
            stem = Path(fname).stem
            clean_stem = _sanitize_for_distribution(stem)
            out_path = os.path.join(output_dir, f"{clean_stem}_mastered.wav")

            if os.path.exists(out_path):
                logger.info(f"[{i}/{len(tracks)}] SKIP (exists): {fname}")
                continue

            logger.info(f"")
            logger.info(f"[{i}/{len(tracks)}] ▶ {fname}")
            t0 = time.time()

            try:
                job_id = submit_job(client, path, out_path)
                logger.info(f"    Job submitted: {job_id}")

                result = poll_job(client, job_id)
                elapsed = time.time() - t0

                dsp_metrics = result.get("result", {}).get("dsp_metrics", {})
                logger.info(
                    f"[{i}/{len(tracks)}] ✓ {fname} → {elapsed:.1f}s | "
                    f"LUFS: {dsp_metrics.get('lufs_after', '?')} | "
                    f"Peak: {dsp_metrics.get('true_peak_after', '?')} dBTP"
                )
                results.append({"file": fname, "elapsed_s": round(elapsed, 1), "metrics": dsp_metrics})

            except Exception as e:
                elapsed = time.time() - t0
                logger.error(f"[{i}/{len(tracks)}] ✗ {fname}: {e} ({elapsed:.1f}s)")
                failed.append({"file": fname, "error": str(e)})

    # Summary
    logger.info(f"")
    logger.info(f"═══════════════════════════════════════════════")
    logger.info(f"  COMPLETE: {len(results)} succeeded, {len(failed)} failed")
    logger.info(f"═══════════════════════════════════════════════")

    if failed:
        for f in failed:
            logger.warning(f"  FAILED: {f['file']}: {f['error']}")

    report_path = os.path.join(output_dir, "_mastering_report.json")
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump({"results": results, "failed": failed}, fp, indent=2, ensure_ascii=False)
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
