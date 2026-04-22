# ===== Stage 1: Build CamillaDSP from source =====
FROM rust:1.85-slim AS camilladsp-builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config libasound2-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

COPY camilladsp/ /build/camilladsp/
WORKDIR /build/camilladsp

# Build CamillaDSP without audio device backends (file I/O only)
# No default features = no websocket, just core DSP
RUN cargo build --release --no-default-features && \
    strip target/release/camilladsp && \
    cp target/release/camilladsp /usr/local/bin/camilladsp

# ===== Stage 2: Python runtime =====
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Copy CamillaDSP binary from builder stage
COPY --from=camilladsp-builder /usr/local/bin/camilladsp /usr/local/bin/camilladsp
RUN chmod +x /usr/local/bin/camilladsp

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rendition_dsp/ ./rendition_dsp/

RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

# Set CamillaDSP binary path
ENV CAMILLADSP_BIN=/usr/local/bin/camilladsp

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "-m", "rendition_dsp"]
