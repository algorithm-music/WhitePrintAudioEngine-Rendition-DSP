"""Entry point for `python -m rendition_dsp`."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "rendition_dsp.main:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
