FROM python:3.12-slim

# Install runtime dependencies + download CamillaDSP prebuilt binary
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 curl && \
    # Download CamillaDSP v4.1.3 prebuilt binary (linux-amd64, ~3MB)
    curl -fsSL https://github.com/HEnquist/camilladsp/releases/download/v4.1.3/camilladsp-linux-amd64.tar.gz \
      | tar -xz -C /usr/local/bin/ && \
    chmod +x /usr/local/bin/camilladsp && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

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
