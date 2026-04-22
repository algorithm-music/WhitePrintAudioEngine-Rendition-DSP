@echo off
REM ═══════════════════════════════════════════════════════════
REM  WhitePrintAudioEngine — Local Full Pipeline Launcher
REM  Starts all 4 services for local mastering:
REM    Audition (8081) → Deliberation (8082) → Rendition DSP (8083) → Concertmaster (8084)
REM ═══════════════════════════════════════════════════════════

set CONCERTMASTER_API_KEY=local-dev-key
set AUDITION_URL=http://localhost:8081
set DELIBERATION_URL=http://localhost:8082
set RENDITION_DSP_URL=http://localhost:8083

echo [1/4] Starting Audition on :8081...
start "Audition" /D "c:\Users\ishij\Documents\GitHub\WhitePrintAudioEngine-Audition" cmd /c "python -m uvicorn audition.main:app --host 0.0.0.0 --port 8081 --log-level info"

echo [2/4] Starting Deliberation on :8082...
start "Deliberation" /D "c:\Users\ishij\Documents\GitHub\WhitePrintAudioEngine-Deliberation" cmd /c "python -m uvicorn deliberation.main:app --host 0.0.0.0 --port 8082 --log-level info"

echo [3/4] Starting Rendition DSP on :8083...
start "Rendition DSP" /D "c:\Users\ishij\Documents\GitHub\WhitePrintAudioEngine-Rendition-DSP" cmd /c "python -m uvicorn rendition_dsp.main:app --host 0.0.0.0 --port 8083 --log-level info"

timeout /t 3 >nul

echo [4/4] Starting Concertmaster on :8084...
start "Concertmaster" /D "c:\Users\ishij\Documents\GitHub\WhitePrintAudioEngine-Concertmaster" cmd /c "set CONCERTMASTER_API_KEY=local-dev-key&& set AUDITION_URL=http://localhost:8081&& set DELIBERATION_URL=http://localhost:8082&& set RENDITION_DSP_URL=http://localhost:8083&& python -m uvicorn concertmaster.main:app --host 0.0.0.0 --port 8084 --log-level info"

echo.
echo All services starting. Wait ~5 seconds then run batch_master_local.py
echo.
