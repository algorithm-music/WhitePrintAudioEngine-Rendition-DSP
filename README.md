# WhitePrintAudioEngine — Rendition DSP

14-Stage Mastering Chain — 忠実実行エンジン。

## 役割

AIが決定したパラメータを**一切の判断なく忠実に実行**する。
ハードコードされたデフォルト値は全てバイパス（何もしない）。
AIが指定しなかったパラメータは加工されない。

## 信号チェーン

```
Input Gain → Transformer Saturation → Triode Drive → Tape Saturation
→ Dynamic EQ → M/S Width → Parallel Drive
→ DC Remove → Parametric EQ (4band) → Compressor → Limiter → Dither
```

## 設計原則

- **デフォルト = バイパス**: 全パラメータのデフォルトは0.0 / 1:1 / OFF
- **判断しない**: ソフトクリップ、リミッター等はAIが有効化しない限りOFF
- **フォールバックなし**: target_lufs / target_true_peak は必須引数（AIが決定）

## API (Internal)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/internal/master-url` | params + audio URL → mastered WAV |
| GET | `/health` | Liveness probe |

## Deploy

```bash
gcloud run deploy whiteprintaudioengine-rendition-dsp \
  --source . --region asia-northeast1 \
  --memory 16Gi --cpu 4 --concurrency 1 --timeout 900 --ingress internal
```

© YOMIBITO SHIRAZU — WhitePrintAudioEngine
