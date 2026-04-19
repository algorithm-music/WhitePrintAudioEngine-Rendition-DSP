# -*- coding: utf-8 -*-
"""
aimastering-rendition-dsp: 14-Stage Mastering Chain as Cloud Run Microservice

Single responsibility: audio bytes + params in → mastered WAV bytes + metrics out.
Stateless. No file storage. No AI calls.
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.responses import Response, JSONResponse, FileResponse
from pydantic import BaseModel, Field
import httpx

from rendition_dsp.services.dsp_engine_v2 import master_audio

FETCH_TIMEOUT = 120.0
MAX_AUDIO_SIZE = 500 * 1024 * 1024  # 500MB

# Use GCSFuse mount for temp files to avoid local memory/disk pressure.
# Falls back to /tmp if GCSFuse mount is not available.
_GCS_TMP = "/mnt/gcs/aimastering-tmp-audio"
TEMP_DIR = _GCS_TMP if os.path.isdir(_GCS_TMP) else None  # None = system default /tmp
if TEMP_DIR:
    os.environ["TMPDIR"] = TEMP_DIR

# ──────────────────────────────────────────
# Logging
# ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("rendition_dsp")


# ──────────────────────────────────────────
# Application Lifecycle
# ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("RENDITION_DSP Engine v2 (14-stage mastering chain) is online.")
    yield
    logger.info("RENDITION_DSP Engine shutting down.")


# ──────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────
app = FastAPI(
    title="rendition_dsp",
    description="14-Stage Analog-Modeled Mastering Chain",
    version="2.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────
# Middleware: Request Tracking
# ──────────────────────────────────────────
@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.monotonic()
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    response = await call_next(request)
    duration_ms = int((time.monotonic() - start_time) * 1000)
    logger.info(f"[{request_id}] Completed in {duration_ms}ms → {response.status_code}")
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Duration-Ms"] = str(duration_ms)
    return response


# ══════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════
class MasterRequest(BaseModel):
    local_path: str = Field(..., description="Absolute path to the input audio file on the mounted volume")
    params: dict = Field(default_factory=dict, description="DSP processing parameters")
    target_lufs: float = Field(default=-14.0)
    target_true_peak: float = Field(default=-1.0)
    output_path: str | None = Field(default=None, description="Optional path to save the output. If not provided, a temp file is returned.")
    output_url: str | None = Field(default=None, description="Optional presigned PUT URL to upload result directly to client's storage")

@app.post("/internal/master")
async def master(
    req: MasterRequest,
    background_tasks: BackgroundTasks,
) -> Response:
    """
    Core endpoint: Apply 14-stage mastering chain to audio.
    Reads input directly from the volume mount, processes, and writes output.
    """
    input_path = req.local_path

    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="File not found on volume.")

    if os.path.getsize(input_path) < 100:
        raise HTTPException(status_code=422, detail="Audio data too small")

    # FUSE workaround: copy to local /tmp to avoid libsndfile SystemError on GCS FUSE mounts
    fd, local_in = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        # copyfile (not copy2) — copy2's copystat fails with EPERM on GCSFuse
        # sources when the container runs as non-root (appuser); only the bytes
        # need to land in /tmp, metadata preservation is irrelevant.
        await asyncio.to_thread(shutil.copyfile, input_path, local_in)
        # Re-assign input_path to purely local file for the duration of the mastering step
        input_path = local_in
    except Exception as copy_err:
        if os.path.exists(local_in):
            os.remove(local_in)
        raise HTTPException(status_code=500, detail=f"Failed to copy input file: {copy_err}")

    output_path = req.output_path
    if not output_path:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TEMP_DIR) as tmp_out:
            output_path = tmp_out.name

    try:
        metrics = await asyncio.to_thread(
            master_audio,
            input_path=input_path,
            output_path=output_path,
            params=req.params,
            target_lufs=req.target_lufs,
            target_true_peak=req.target_true_peak,
        )
    except Exception as e:
        logger.error(f"Mastering failed: {type(e).__name__}: {e}")
        if req.output_path is None and os.path.exists(output_path):
            os.remove(output_path)
        raise HTTPException(
            status_code=500,
            detail="Mastering failed. Check server logs for details.",
        )
    finally:
        if os.path.exists(local_in):
            os.remove(local_in)

    # Direct push to client storage if requested
    if req.output_url:
        try:
            async def file_streamer(path):
                with open(path, "rb") as f_out:
                    while chunk := f_out.read(65536):
                        yield chunk

            async with httpx.AsyncClient(timeout=FETCH_TIMEOUT) as client:
                res = await client.put(
                    req.output_url, 
                    content=file_streamer(output_path),
                    headers={"Content-Type": "audio/wav"}
                )
                res.raise_for_status()
                
            logger.info("Successfully pushed mastered audio to client storage mapping.")
            
            if req.output_path is None and os.path.exists(output_path):
                os.remove(output_path)
                
            return JSONResponse(content={
                "status": "success",
                "message": "Audio pushed to output_url",
                "metrics": metrics
            })
            
        except Exception as e:
            logger.error(f"Failed to push to output_url: {e}")
            if req.output_path is None and os.path.exists(output_path):
                os.remove(output_path)
            raise HTTPException(status_code=502, detail=f"Failed to push to output_url: {e}")

    # If the user specified an output path, return JSON with path and metrics instead of streaming the FileResponse
    if req.output_path:
        return JSONResponse(content={
            "status": "success",
            "output_path": output_path,
            "metrics": metrics
        })

    # Cleanup output file after response is sent
    background_tasks.add_task(lambda p: os.remove(p) if os.path.exists(p) else None, output_path)

    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        headers={"X-Metrics": json.dumps(metrics)},
    )




class MasterUrlRequest(BaseModel):
    audio_url: str = Field(..., description="Direct download URL for audio file")
    output_url: str | None = Field(default=None, description="Optional presigned PUT URL to upload result directly to client's storage")
    params: dict = Field(default_factory=dict, description="RENDITION_DSP parameter dict")
    target_lufs: float = Field(default=-14.0)
    target_true_peak: float = Field(default=-1.0)


@app.post("/internal/master-url")
async def master_url(req: MasterUrlRequest, background_tasks: BackgroundTasks) -> Response:
    """
    URL-based mastering: 
    Streams audio input to disk safely, processes it, and streams it back.
    If output_url is provided, it streams the output directly to the client's storage and returns metrics.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TEMP_DIR) as tmp_in:
        input_path = tmp_in.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TEMP_DIR) as tmp_out:
        output_path = tmp_out.name

    # Stream download from URL
    try:
        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT, follow_redirects=True, max_redirects=5,
        ) as client:
            async with client.stream("GET", req.audio_url) as resp:
                resp.raise_for_status()
                with open(input_path, "wb") as f_in:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        f_in.write(chunk)
    except httpx.HTTPStatusError as e:
        os.remove(input_path)
        os.remove(output_path)
        raise HTTPException(status_code=502, detail=f"Failed to fetch audio: {e.response.status_code}")
    except httpx.TimeoutException:
        os.remove(input_path)
        os.remove(output_path)
        raise HTTPException(status_code=504, detail="Audio download timed out")
    except Exception as e:
        os.remove(input_path)
        os.remove(output_path)
        raise HTTPException(status_code=502, detail=f"Audio fetch error: {type(e).__name__}: {e}")

    input_size = os.path.getsize(input_path)
    if input_size > MAX_AUDIO_SIZE:
        os.remove(input_path)
        os.remove(output_path)
        raise HTTPException(status_code=413, detail=f"Audio too large: {input_size / 1024 / 1024:.0f}MB (max 500MB)")
    if input_size < 100:
        os.remove(input_path)
        os.remove(output_path)
        raise HTTPException(status_code=422, detail="Audio data too small")

    logger.info(f"Fetched {input_size / 1024 / 1024:.1f}MB from URL to local disk")

    # Run RENDITION_DSP chain via disk (non-blocking)
    try:
        metrics = await asyncio.to_thread(master_audio,
            input_path=input_path,
            output_path=output_path,
            params=req.params,
            target_lufs=req.target_lufs,
            target_true_peak=req.target_true_peak,
        )
    except Exception as e:
        logger.error(f"Mastering failed: {type(e).__name__}: {e}")
        os.remove(input_path)
        os.remove(output_path)
        raise HTTPException(
            status_code=500,
            detail="Mastering failed. Check server logs for details.",
        )
    finally:
        # Cleanup input file
        if os.path.exists(input_path):
            os.remove(input_path)

    # Direct push to client storage if requested
    if req.output_url:
        try:
            async def file_streamer(path):
                with open(path, "rb") as f_out:
                    while chunk := f_out.read(65536):
                        yield chunk

            async with httpx.AsyncClient(timeout=FETCH_TIMEOUT) as client:
                res = await client.put(
                    req.output_url, 
                    content=file_streamer(output_path),
                    headers={"Content-Type": "audio/wav"}
                )
                res.raise_for_status()
                
            logger.info("Successfully pushed mastered audio to client storage mapping.")
            
            # Clean up immediately since we're done with the file
            if os.path.exists(output_path):
                os.remove(output_path)
                
            return JSONResponse(content={
                "status": "success",
                "message": "Audio pushed to output_url",
                "metrics": metrics
            })
            
        except Exception as e:
            logger.error(f"Failed to push to output_url: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            raise HTTPException(status_code=502, detail=f"Failed to push to output_url: {e}")

    # No output_url: return FileResponse streaming
    background_tasks.add_task(lambda p: os.remove(p) if os.path.exists(p) else None, output_path)
    
    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        headers={"X-Metrics": json.dumps(metrics)},
    )


@app.get("/")
async def index() -> JSONResponse:
    """Root endpoint providing service identity."""
    return JSONResponse(content={
        "status": "online",
        "service": "rendition_dsp",
        "engine": "14-Stage Analog-Modeled Mastering Chain",
        "message": "Audio DSP rendering microservice is ready.",
        "documentation": "/docs"
    })


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={
        "status": "ready",
        "service": "rendition_dsp",
        "version": "2.0.0",
        "engine": "14-Stage Analog-Modeled Mastering Chain",
        "stages": 14,
        "stores_audio": False,
    })
