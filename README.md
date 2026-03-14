# WhitePrintAudioEngine — Rendition DSP

14-Stage Mastering Chain — The final rendering engine.

## API (Internal)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/internal/master-url` | params + audio URL → mastered WAV |
| GET | `/health` | Liveness probe |

## Deploy

```bash
gcloud run deploy aimastering-rendition-dsp \
  --source . --region asia-northeast1 \
  --memory 2Gi --cpu 2 --concurrency 1 --timeout 600 --ingress internal
```

© YOMIBITO SHIRAZU — WhitePrintAudioEngine
