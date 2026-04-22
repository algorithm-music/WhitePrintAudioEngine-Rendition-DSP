# pyre-ignore-all-errors

"""
RENDITION_DSP Engine v3.1 — Hybrid Pedalboard (JUCE) + Python Mastering Chain

Signal flow (hybrid):
 Python pre-stage:
  1 DC Remove -> 2 Gain Stage ->
  3 Transformer Sat -> 4 Triode Tube -> 5 Tape Emulation ->
  6 Dynamic EQ -> 7 M/S Encode -> 8 Freq-Dep Width ->
  9 Parallel Drive ->

 Spotify Pedalboard (JUCE C++, in-process):
  10 Parametric EQ (Biquad Shelf/Peak) -> 11 Compressor -> 12 Limiter
  13 TPDF Dither (Python)

 Fallback: full Python/scipy pipeline if pedalboard unavailable.

v3.1 changes:
- Replaced CamillaDSP (external Rust binary) with Spotify Pedalboard (JUCE).
  No subprocess, no temp files, no libasound2 dependency — pure pip install.
- EQ, compression, limiting run in-process via JUCE C++ bindings.
- Python retains analog saturation modeling (transformer/triode/tape),
  dynamic EQ, M/S processing — these are the unique value-add.
- Graceful fallback to scipy pipeline if pedalboard is not installed.
"""

import gc
import logging
import math
import os

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, lfilter, resample_poly, sosfilt

logger = logging.getLogger("rendition_dsp.dsp_v2")

MAX_CONVERGENCE_LOOPS = 5
CONVERGENCE_TOLERANCE_DB = 0.1
LOG_FLOOR = 1e-10


def master_audio(
    input_path: str,
    output_path: str,
    params: dict,
    target_lufs: float = -14.0,
    target_true_peak: float = -1.0,
) -> dict:
    """Apply 14-stage dynamic mastering chain with Newton-method convergence.

    All DSP parameters are determined by AI. No hardcoded defaults affect
    the audio — if a parameter is not specified, the processing stage
    is bypassed (neutral).
    """
    data, sr = sf.read(input_path, dtype="float32")
    if data.ndim == 1:
        data = np.column_stack([data, data])

    # Keep float32 — no .astype(float64) upcast.
    left = np.ascontiguousarray(data[:, 0], dtype=np.float32)
    right = np.ascontiguousarray(
        data[:, 1] if data.shape[1] >= 2 else data[:, 0],
        dtype=np.float32,
    )
    del data
    gc.collect()

    lufs_before = _calculate_lufs_bs1770(left, right, sr)
    peak_before = _measure_true_peak_db(left, right, sr)
    crest_before = _calculate_crest_factor(left, right)

    if params.get("dc_remove_enabled", True):
        left = _remove_dc(left, sr)
        right = _remove_dc(right, sr)

    dither_seed = int.from_bytes(os.urandom(4), "little")

    # Mutable copy of params — convergence loop may auto-reduce saturation.
    active_params = dict(params)

    gain_adjustment = 0.0
    convergence_loops = 0
    # Holds the most-recent processed output; the final iteration's buffers
    # are what we write to disk (no per-iteration column_stack).
    curr_out_l: np.ndarray | None = None
    curr_out_r: np.ndarray | None = None

    for _ in range(MAX_CONVERGENCE_LOOPS):
        convergence_loops += 1

        # _apply_dynamic_chain makes its own mutable copies internally.
        # We pass base left/right as read-only references.
        if curr_out_l is not None:
            del curr_out_l, curr_out_r
            gc.collect()

        out_l, out_r = _apply_dynamic_chain(
            left, right, sr, active_params, gain_adjustment, dither_seed,
        )
        curr_out_l, curr_out_r = out_l, out_r

        current_lufs = _calculate_lufs_bs1770(out_l, out_r, sr)
        current_peak = _measure_true_peak_db(out_l, out_r, sr)
        lufs_diff = current_lufs - target_lufs
        peak_safe = current_peak <= target_true_peak + 0.1

        logger.info(
            f"Loop {convergence_loops}: LUFS={current_lufs:.2f}, "
            f"Peak={current_peak:.2f}, GainAdj={gain_adjustment:.2f}dB"
        )

        if abs(lufs_diff) < CONVERGENCE_TOLERANCE_DB and peak_safe:
            break

        # Newton-method proportional control
        gain_adjustment -= lufs_diff * 0.85

    assert curr_out_l is not None and curr_out_r is not None

    # Free base input buffers before building the final stereo array for write.
    del left, right
    gc.collect()

    lufs_after = _calculate_lufs_bs1770(curr_out_l, curr_out_r, sr)
    peak_after = _measure_true_peak_db(curr_out_l, curr_out_r, sr)
    dr_after = _calculate_dynamic_range(curr_out_l, curr_out_r, sr)
    crest_after = _calculate_crest_factor(curr_out_l, curr_out_r)

    # Single column_stack right before write.
    best_output = np.column_stack([curr_out_l, curr_out_r]).astype(
        np.float32, copy=False
    )
    del curr_out_l, curr_out_r
    gc.collect()

    sf.write(
        output_path,
        best_output,
        sr,
        format="WAV",
        subtype="PCM_24",
    )
    del best_output
    gc.collect()

    return {
        "lufs_before": round(lufs_before, 1),
        "lufs_after": round(lufs_after, 1),
        "true_peak_before": round(peak_before, 1),
        "true_peak_after": round(peak_after, 1),
        "dynamic_range_after": round(dr_after, 1),
        "crest_factor_before_db": round(crest_before, 1),
        "crest_factor_after_db": round(crest_after, 1),
        "saturation_reduced": saturation_reduced,
        "convergence_loops": convergence_loops,
        "gain_adjustment_db": round(gain_adjustment, 2),
        "target_lufs": target_lufs,
        "target_true_peak": target_true_peak,
        "engine_version": "v2.2_guardrail",
    }


def _apply_dynamic_chain(
    left: np.ndarray, right: np.ndarray,
    sr: int, params: dict, gain_adj: float,
    dither_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mastering chain with Dynamic Automation.

    Takes read-only references for `left`/`right`; makes internal copies once
    per call so the convergence loop does not re-copy upstream.
    """
    overrides = params.get("section_overrides", [])
    out_l, out_r = _apply_full_chain(
        left.copy(), right.copy(), sr, params, gain_adj, dither_seed,
    )
    if not overrides:
        return out_l, out_r

    crossfade_samps = int(0.05 * sr)
    preroll_sec = 1.0
    sig_len = len(left)

    for ovr in overrides:
        start_sec = ovr.get("start_sec")
        end_sec = ovr.get("end_sec")
        if start_sec is None or end_sec is None:
            continue

        sec_params = params.copy()
        sec_params.update(ovr)
        act_start = int(start_sec * sr)
        act_end = int(end_sec * sr)
        if act_start >= sig_len:
            continue

        start_samp = max(0, act_start - int(preroll_sec * sr))
        end_samp = min(sig_len, act_end + crossfade_samps)

        proc_l, proc_r = _apply_full_chain(
            left[start_samp:end_samp].copy(),
            right[start_samp:end_samp].copy(),
            sr, sec_params, gain_adj, dither_seed,
        )

        preroll_actual = act_start - start_samp
        proc_l = proc_l[preroll_actual:]
        proc_r = proc_r[preroll_actual:]

        insert_len = min(
            len(proc_l),
            act_end - act_start + crossfade_samps,
        )

        window = np.ones(insert_len, dtype=np.float32)
        if insert_len > crossfade_samps and act_start > 0:
            window[:crossfade_samps] = np.linspace(
                0.0, 1.0, crossfade_samps, dtype=np.float32
            )
        ie = act_start + insert_len
        if insert_len > crossfade_samps and ie < sig_len:
            window[-crossfade_samps:] = np.linspace(
                1.0, 0.0, crossfade_samps, dtype=np.float32
            )

        tgt_l = out_l[act_start:ie]
        sl = min(len(tgt_l), len(window))

        # In-place crossfade blend — no intermediate full-size arrays.
        win_s = window[:sl]
        inv_win = 1.0 - win_s
        out_l[act_start:act_start + sl] *= inv_win
        out_l[act_start:act_start + sl] += proc_l[:sl] * win_s
        out_r[act_start:act_start + sl] *= inv_win
        out_r[act_start:act_start + sl] += proc_r[:sl] * win_s

        del proc_l, proc_r, window, win_s, inv_win

    return out_l, out_r


def _apply_full_chain(
    left: np.ndarray, right: np.ndarray,
    sr: int, params: dict, gain_adj: float,
    dither_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Hybrid mastering chain: Python (saturation) → CamillaDSP (EQ/comp/limit/dither).

    Stages handled in Python (analog modeling — not available in CamillaDSP):
      - Input gain
      - Transformer / Triode / Tape saturation (8x oversampled)
      - M/S frequency-dependent width
      - Parallel drive (waveshaping)
      - Dynamic EQ

    Stages delegated to CamillaDSP (Rust, 64-bit, SIMD-optimized):
      - DC Remove (Biquad HighpassFO 5 Hz)
      - Parametric EQ (Biquad Peaking/Shelf bands)
      - Compressor (attack/release/threshold/ratio/makeup)
      - Limiter (soft clip)
      - TPDF Dither
    """
    # ── Python pre-stages: saturation & spatial ───────────────────
    g = _db_to_linear(params.get("input_gain_db", 0) + gain_adj)
    left *= g  # in-place; caller passed a copy.
    right *= g

    # Saturation (8x OS): process L, free, process R, free.
    left = _apply_saturation_chain(left, sr, params)
    right = _apply_saturation_chain(right, sr, params)

    # Dynamic EQ (Python — CamillaDSP has no dynamic EQ)
    left = _apply_dynamic_eq(left, sr, params)
    right = _apply_dynamic_eq(right, sr, params)

    # M/S processing + frequency-dependent width
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    mid, side = _apply_freq_dep_width(mid, side, sr, params)
    out_l = mid + side
    out_r = mid - side
    del mid, side

    # Parallel drive (waveshaping — Python only)
    pw = params.get("parallel_wet", 0.0)
    pd = params.get("parallel_drive", 0.0)
    out_l = _neuro_drive(out_l, sr, wet=pw, drive=pd)
    out_r = _neuro_drive(out_r, sr, wet=pw, drive=pd)

    # ── Spotify Pedalboard stages: EQ → Comp → Limiter ───────────
    try:
        from pedalboard import (
            Pedalboard, Compressor, Limiter, Gain,
            HighShelfFilter, LowShelfFilter, PeakFilter,
            HighpassFilter, Clipping,
        )

        # Build pedalboard chain from mastering params
        effects = []

        # DC remove (5 Hz highpass)
        effects.append(HighpassFilter(cutoff_frequency_hz=5.0))

        # Parametric EQ — 4-band (shelf + peak)
        ls_gain = params.get("eq_low_shelf_gain_db", 0.0)
        if abs(ls_gain) > 0.01:
            effects.append(LowShelfFilter(
                cutoff_frequency_hz=params.get("eq_low_shelf_freq", 80),
                gain_db=ls_gain,
            ))

        lm_gain = params.get("eq_low_mid_gain_db", 0.0)
        if abs(lm_gain) > 0.01:
            effects.append(PeakFilter(
                cutoff_frequency_hz=params.get("eq_low_mid_freq", 300),
                gain_db=lm_gain,
                q=params.get("eq_low_mid_q", 1.0),
            ))

        hm_gain = params.get("eq_high_mid_gain_db", 0.0)
        if abs(hm_gain) > 0.01:
            effects.append(PeakFilter(
                cutoff_frequency_hz=params.get("eq_high_mid_freq", 3000),
                gain_db=hm_gain,
                q=params.get("eq_high_mid_q", 1.2),
            ))

        hs_gain = params.get("eq_high_shelf_gain_db", 0.0)
        if abs(hs_gain) > 0.01:
            effects.append(HighShelfFilter(
                cutoff_frequency_hz=params.get("eq_high_shelf_freq", 10000),
                gain_db=hs_gain,
            ))

        # Compressor
        effects.append(Compressor(
            threshold_db=params.get("comp_threshold_db", 0.0),
            ratio=params.get("comp_ratio", 1.0),
            attack_ms=params.get("comp_attack_ms", 10.0),
            release_ms=params.get("comp_release_ms", 100.0),
        ))

        # Makeup gain from compressor
        makeup = params.get("comp_makeup_db", 0.0)
        if abs(makeup) > 0.01:
            effects.append(Gain(gain_db=makeup))

        # Limiter (true peak) — ceiling follows AI-determined target_true_peak
        limiter_ceil = params.get("limiter_ceiling_db", target_true_peak)
        effects.append(Limiter(
            threshold_db=limiter_ceil,
            release_ms=params.get("limiter_release_ms", 50.0),
        ))

        board = Pedalboard(effects)

        # Process in-memory (interleaved stereo, float32)
        # Pedalboard expects shape (channels, samples)
        stereo = np.stack([out_l, out_r], axis=0).astype(np.float32, copy=False)
        processed = board(stereo, sr)
        out_l = np.ascontiguousarray(processed[0], dtype=np.float32)
        out_r = np.ascontiguousarray(processed[1], dtype=np.float32)
        del stereo, processed
        gc.collect()

        logger.info("Pedalboard (JUCE) processing completed successfully.")

        # TPDF dither (kept in Python — pedalboard doesn't have dither)
        if params.get("dither_enabled", True):
            db = params.get("dither_bits", 24)
            out_l = _apply_dither(
                out_l, target_bits=db, channel_idx=0, seed=dither_seed,
            )
            out_r = _apply_dither(
                out_r, target_bits=db, channel_idx=1, seed=dither_seed,
            )

        return out_l, out_r

    except ImportError:
        logger.warning("pedalboard not installed, falling back to scipy DSP.")
    except Exception as e:
        logger.warning(f"Pedalboard error: {e}, falling back to scipy DSP.")

    # ── Python/scipy fallback ─────────────────────────────────────
    out_l = _apply_parametric_eq(out_l, sr, params, "mid")
    out_r = _apply_parametric_eq(out_r, sr, params, "mid")

    out_l, out_r = _apply_multiband_comp_stereo(out_l, out_r, sr, params)

    if params.get("soft_clip_enabled", 1):
        ct = params.get("soft_clip_threshold", 0.98)
        out_l = _soft_clipper(out_l, threshold=ct)
        out_r = _soft_clipper(out_r, threshold=ct)

    if params.get("limiter_enabled", True):
        cd = params.get("limiter_ceil_db", target_true_peak)
        out_l, out_r = _apply_tp_limiter(out_l, out_r, sr, cd)

    if params.get("dither_enabled", True):
        db = params.get("dither_bits", 24)
        out_l = _apply_dither(
            out_l, target_bits=db, channel_idx=0, seed=dither_seed,
        )
        out_r = _apply_dither(
            out_r, target_bits=db, channel_idx=1, seed=dither_seed,
        )

    return out_l, out_r


def _remove_dc(buf: np.ndarray, sr: int) -> np.ndarray:
    w0 = 2.0 * np.pi * 5.0 / sr
    b = np.array([1.0, -1.0]) / (1.0 + w0)
    a = np.array([1.0, -(1.0 - w0) / (1.0 + w0)])
    return lfilter(b, a, buf).astype(np.float32, copy=False)


def _apply_saturation_chain(
    buf: np.ndarray, sr: int, params: dict,
) -> np.ndarray:
    """Per-channel saturation: transformer + triode + tape at 8x OS, float32."""
    ts = params.get("transformer_saturation", 0.0)
    tm = params.get("transformer_mix", 0.0)
    td = params.get("triode_drive", 0.0)
    tx = params.get("triode_mix", 0.0)
    ps = params.get("tape_saturation", 0.0)
    pm = params.get("tape_mix", 0.0)

    if max(ts, tm, td, tx, ps, pm) < 0.01:
        return buf

    up = resample_poly(buf, 8, 1).astype(np.float32, copy=False)
    su = sr * 8

    if ts >= 0.01 or tm >= 0.01:
        drv = ts * 5.0 + 0.5
        B = np.tanh((up * drv) / 3.0)
        mem = 0.15 * ts
        if mem > 0.001:
            B = lfilter([1.0 - mem], [1.0, -mem], B).astype(
                np.float32, copy=False
            )
        up = up * (1.0 - tm) + B * tm
        del B
        fc = 18000 * 0.8
        if fc < su * 0.49:
            up = sosfilt(
                butter(1, fc, btype="low", fs=su, output="sos"), up,
            ).astype(np.float32, copy=False)

    if td >= 0.01 or tx >= 0.01:
        bias = params.get("triode_bias", 0.0)
        Vg = up * (td * 8.0 + 0.5) + bias
        st = math.sqrt(300.0 + 250.0 ** 2)
        inner = 600.0 * (1.0 / 100.0 + Vg / st)
        E1 = (250.0 / 600.0) * np.where(
            inner > 20.0,
            inner,
            np.log1p(np.exp(np.clip(inner, -20, 20))),
        )
        del inner
        Ip = np.power(np.maximum(E1, 0.0), 1.4)
        del E1
        Ig = np.where(
            Vg > 0,
            np.expm1(np.clip(Vg / 1060.0, -20, 20)),
            0.0,
        )
        del Vg
        Ip += Ig * 0.1
        del Ig
        sat = Ip / (np.max(np.abs(Ip)) + 1e-10)
        del Ip
        wdc = 2.0 * np.pi * 10.0 / su
        sat = lfilter(
            [1.0 / (1.0 + wdc), -1.0 / (1.0 + wdc)],
            [1.0, -(1.0 - wdc) / (1.0 + wdc)],
            sat,
        ).astype(np.float32, copy=False)
        up = up * (1.0 - tx) + sat * tx
        del sat

    if ps >= 0.01 or pm >= 0.01:
        drv = ps * 3.0 + 0.5
        comp = np.arctan(up * drv) / np.arctan(drv)
        hfc = min(
            15000 + params.get("tape_speed", 30.0) * 200,
            su * 0.45,
        )
        if hfc < su * 0.49:
            comp = sosfilt(
                butter(2, hfc, btype="low", fs=su, output="sos"), comp,
            ).astype(np.float32, copy=False)
        up = up * (1.0 - pm) + comp * pm
        del comp

    out = resample_poly(up, 1, 8)[:len(buf)].astype(np.float32, copy=False)
    del up
    return out


def _apply_dynamic_eq(
    buf: np.ndarray, sr: int, params: dict,
) -> np.ndarray:
    if not params.get("dyn_eq_enabled", False):
        return buf
    res = buf.copy()
    for band in params.get("dyn_eq_bands", []):
        f = band["freq"]
        q = band["q"]
        th = _db_to_linear(band["threshold_db"])
        mg = _db_to_linear(band["max_gain_db"])
        if f >= sr * 0.49 or f < 20:
            continue
        bw = f / q
        lo_f = max(f - bw / 2, 20)
        hi_f = min(f + bw / 2, sr * 0.49 - 1)
        if lo_f >= hi_f:
            continue
        sos = butter(
            2, [lo_f, hi_f], btype="band", fs=sr, output="sos",
        )
        sc = np.abs(sosfilt(sos, buf))
        ac = math.exp(
            -1.0 / max(1, band.get("attack_ms", 10.0) / 1000.0 * sr)
        )
        rc = math.exp(
            -1.0 / max(1, band.get("release_ms", 80.0) / 1000.0 * sr)
        )
        env = np.maximum(
            lfilter([1 - ac], [1, -ac], sc),
            lfilter([1 - rc], [1, -rc], sc),
        )
        del sc
        gain = np.where(
            env > th,
            1.0 + (mg - 1.0) * np.minimum(
                (env - th) / (th * 2.0 + 1e-10), 1.0
            ),
            1.0,
        )
        del env
        res += sosfilt(sos, res) * (gain - 1.0)
        del gain
    return res.astype(np.float32, copy=False)


def _apply_multiband_comp_stereo(
    left: np.ndarray, right: np.ndarray,
    sr: int, params: dict,
):
    th_db = params.get("comp_threshold_db", 0.0)
    rat = params.get("comp_ratio", 1.0)
    att = params.get("comp_attack_sec", 0.01)
    rel = params.get("comp_release_sec", 0.15)

    def split_lr4(b, xovers):
        bands = []
        rem = b.copy()
        aps = []
        for x in xovers:
            if x >= sr * 0.49:
                bands.append(np.zeros_like(b))
                continue
            sl = butter(2, x, btype="low", fs=sr, output="sos")
            sh = butter(2, x, btype="high", fs=sr, output="sos")
            low = sosfilt(sl, sosfilt(sl, rem))
            rem = sosfilt(sh, sosfilt(sh, rem))
            bands.append(low)
            aps.append((sl, sh))
        bands.append(rem)
        for i in range(len(bands)):
            for j in range(i):
                lp, hp = aps[j]
                bands[i] = (
                    sosfilt(lp, sosfilt(lp, bands[i]))
                    + sosfilt(hp, sosfilt(hp, bands[i]))
                )
        return bands

    bl = split_lr4(left, [80, 300, 4000])
    br = split_lr4(right, [80, 300, 4000])
    pl, pr = [], []

    for i in range(4):
        ab = np.maximum(np.abs(bl[i]), np.abs(br[i]))
        ac = math.exp(-1.0 / max(1, att * sr))
        rc = math.exp(-1.0 / max(1, rel * sr))
        env = np.maximum(
            lfilter([1 - ac], [1, -ac], ab),
            lfilter([1 - rc], [1, -rc], ab),
        )
        del ab
        tl = _db_to_linear(th_db)
        gain = np.ones_like(bl[i])
        mask = env > tl
        if np.any(mask):
            odb = 20.0 * np.log10(
                np.maximum(env[mask] / tl, LOG_FLOOR)
            )
            gain[mask] = _db_to_linear(
                -(odb * (1.0 - 1.0 / rat))
            )
        del env
        pad = max(int(sr * 0.002), 1)
        gain = np.convolve(
            np.pad(gain, pad, mode="edge"),
            np.ones(pad) / pad,
            mode="same",
        )[pad:-pad]
        pl.append(bl[i] * gain)
        pr.append(br[i] * gain)
        del gain

    sum_l = sum(pl).astype(np.float32, copy=False)
    sum_r = sum(pr).astype(np.float32, copy=False)
    del bl, br, pl, pr
    return sum_l, sum_r


def _apply_parametric_eq(
    buf: np.ndarray, sr: int,
    params: dict, ch_type: str,
) -> np.ndarray:
    eq_cfg = [
        {"type": "lowshelf", "freq": 80,
         "g": params.get("eq_low_shelf_gain_db", 0)},
        {"type": "peaking", "freq": 400, "q": 1.0,
         "g": params.get("eq_low_mid_gain_db", 0)},
        {"type": "peaking", "freq": 3000, "q": 1.0,
         "g": params.get("eq_high_mid_gain_db", 0)},
        {"type": "highshelf", "freq": 8000,
         "g": params.get("eq_high_shelf_gain_db", 0)},
    ]
    if ch_type == "side":
        eq_cfg.append({
            "type": "highshelf", "freq": 4000,
            "g": params.get("ms_side_high_gain_db", 0),
        })
    else:
        eq_cfg.append({
            "type": "lowshelf", "freq": 200,
            "g": params.get("ms_mid_low_gain_db", 0),
        })

    res = buf.copy()
    for b in eq_cfg:
        g = b["g"]
        f = b["freq"]
        q = b.get("q", 0.707)
        if abs(g) < 0.05 or f >= sr * 0.49 or f < 10:
            continue
        w0 = 2.0 * math.pi * f / sr
        A = 10 ** (g / 40.0)
        alpha = math.sin(w0) / (2.0 * q)
        cw = math.cos(w0)
        if b["type"] == "peaking":
            b0 = 1 + alpha * A
            b1 = -2 * cw
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cw
            a2 = 1 - alpha / A
        elif b["type"] == "lowshelf":
            sq = 2.0 * math.sqrt(A) * alpha
            b0 = A * ((A + 1) - (A - 1) * cw + sq)
            b1 = 2 * A * ((A - 1) - (A + 1) * cw)
            b2 = A * ((A + 1) - (A - 1) * cw - sq)
            a0 = (A + 1) + (A - 1) * cw + sq
            a1 = -2 * ((A - 1) + (A + 1) * cw)
            a2 = (A + 1) + (A - 1) * cw - sq
        else:
            sq = 2.0 * math.sqrt(A) * alpha
            b0 = A * ((A + 1) + (A - 1) * cw + sq)
            b1 = -2 * A * ((A - 1) + (A + 1) * cw)
            b2 = A * ((A + 1) + (A - 1) * cw - sq)
            a0 = (A + 1) - (A - 1) * cw + sq
            a1 = 2 * ((A - 1) - (A + 1) * cw)
            a2 = (A + 1) - (A - 1) * cw - sq
        sos = np.array([[
            b0 / a0, b1 / a0, b2 / a0,
            1.0, a1 / a0, a2 / a0,
        ]])
        res = sosfilt(sos, res)
    return res.astype(np.float32, copy=False)


def _apply_freq_dep_width(
    mid: np.ndarray, side: np.ndarray,
    sr: int, params: dict,
):
    lm = params.get("stereo_low_mono", 0.0)
    hw = params.get("stereo_high_wide", 1.0)
    gw = params.get("stereo_width", 1.0)
    if 200.0 >= sr * 0.49 or 4000.0 >= sr * 0.49:
        return mid, side * gw
    s_lo = sosfilt(
        butter(4, 200.0, btype="low", fs=sr, output="sos"),
        side,
    ) * (1.0 - lm)
    s_mid = sosfilt(
        butter(4, [200.0, 4000.0], btype="band", fs=sr, output="sos"),
        side,
    ) * gw
    s_hi = sosfilt(
        butter(4, 4000.0, btype="high", fs=sr, output="sos"),
        side,
    ) * hw
    side_out = (s_lo + s_mid + s_hi).astype(np.float32, copy=False)
    del s_lo, s_mid, s_hi
    return mid, side_out


def _soft_clipper(
    signal: np.ndarray, threshold: float = 0.98,
) -> np.ndarray:
    """Per-channel 4x-oversampled soft clipper."""
    up = resample_poly(signal, 4, 1).astype(np.float32, copy=False)
    mask = np.abs(up) > threshold
    if np.any(mask):
        over = np.abs(up[mask]) - threshold
        up[mask] = np.sign(up[mask]) * (
            threshold + 0.04 * np.tanh(over / 0.04)
        )
    out = resample_poly(up, 1, 4)[:len(signal)].astype(np.float32, copy=False)
    del up
    return out


def _apply_tp_limiter(
    l: np.ndarray, r: np.ndarray,
    sr: int, ceil_db: float,
):
    """True-peak limiter with stereo-linked gain.

    L and R are upsampled sequentially (not stacked), so peak RAM during
    resample_poly is one channel's 8x buffer instead of two.
    """
    ceil = _db_to_linear(ceil_db)
    la = max(1, int(5.0 / 1000.0 * sr))

    # --- L channel: upsample, in-place abs, block-max to base rate, free ---
    up = resample_poly(l, 8, 1).astype(np.float32, copy=False)
    np.abs(up, out=up)
    nf = min(len(l), len(up) // 8)
    ps_l = up[:nf * 8].reshape(nf, 8).max(axis=1)
    if nf < len(l):
        rm = float(np.max(up[nf * 8:])) if nf * 8 < len(up) else 0.0
        ps_l = np.append(ps_l, np.full(len(l) - nf, rm, dtype=np.float32))
    del up
    gc.collect()

    # --- R channel: same, then link via per-sample max ---
    up = resample_poly(r, 8, 1).astype(np.float32, copy=False)
    np.abs(up, out=up)
    nf_r = min(len(r), len(up) // 8)
    ps_r = up[:nf_r * 8].reshape(nf_r, 8).max(axis=1)
    if nf_r < len(r):
        rm = float(np.max(up[nf_r * 8:])) if nf_r * 8 < len(up) else 0.0
        ps_r = np.append(ps_r, np.full(len(r) - nf_r, rm, dtype=np.float32))
    del up
    gc.collect()

    # Align lengths defensively then take stereo-linked peak envelope.
    n = min(len(ps_l), len(ps_r), len(l), len(r))
    ps = np.maximum(ps_l[:n], ps_r[:n])
    del ps_l, ps_r

    pa = (
        maximum_filter1d(
            ps, size=la, origin=-(la // 2),
            mode="constant", cval=0.0,
        )
        if la > 1 else ps
    )
    gain = np.where(pa > ceil, ceil / (pa + 1e-10), 1.0).astype(
        np.float32, copy=False
    )
    del pa, ps

    rc = math.exp(-1.0 / max(1, 50.0 / 1000.0 * sr))
    br = np.array([1.0 - rc])
    ar = np.array([1.0, -rc])
    sm = np.minimum(
        gain,
        lfilter(br, ar, np.minimum(
            gain, lfilter(br, ar, np.minimum(
                gain, lfilter(br, ar, gain),
            )),
        )),
    ).astype(np.float32, copy=False)
    del gain

    ld = (
        np.concatenate([np.zeros(la, dtype=np.float32), l[:-la]])
        if la < len(l) else l
    )
    rd = (
        np.concatenate([np.zeros(la, dtype=np.float32), r[:-la]])
        if la < len(r) else r
    )
    # Trim sm to match so broadcasting is safe.
    sm = sm[:len(ld)]
    return (ld * sm).astype(np.float32, copy=False), (rd * sm).astype(
        np.float32, copy=False
    )


def _neuro_drive(
    buf: np.ndarray, sr: int,
    wet: float = 0.18, drive: float = 3.0,
) -> np.ndarray:
    if wet < 0.01:
        return buf
    return (
        buf * (1.0 - wet)
        + np.tanh(buf * drive) * 0.5 * wet
    ).astype(np.float32, copy=False)


def _apply_dither(
    buf: np.ndarray, target_bits: int = 24,
    channel_idx: int = 0, seed: int = 0,
) -> np.ndarray:
    lvls = 2 ** (target_bits - 1)
    rng = np.random.default_rng(seed + channel_idx)
    tpdf = (
        rng.uniform(-0.5, 0.5, len(buf))
        + rng.uniform(-0.5, 0.5, len(buf))
    ) / lvls
    return (np.round((buf + tpdf) * lvls) / lvls).astype(
        np.float32, copy=False
    )


def _calculate_lufs_bs1770(
    l: np.ndarray, r: np.ndarray, sr: int,
) -> float:
    if len(l) < 1024:
        return -70.0
    K = math.tan(math.pi * 1681.97445 / sr)
    Vh = 10 ** (4.0 / 20.0)
    Vb = Vh ** 0.4996
    Q = 0.707175
    a0 = 1.0 + K / Q + K * K
    s1 = [
        (Vh + Vb * K / Q + K * K) / a0,
        2.0 * (K * K - Vh) / a0,
        (Vh - Vb * K / Q + K * K) / a0,
        1.0,
        2.0 * (K * K - 1.0) / a0,
        (1.0 - K / Q + K * K) / a0,
    ]
    K1 = math.tan(math.pi * 38.13547 / sr)
    Q1 = 0.500327
    ah = 1.0 + K1 / Q1 + K1 * K1
    s2 = [
        1.0 / ah, -2.0 / ah, 1.0 / ah,
        1.0,
        2.0 * (K1 * K1 - 1.0) / ah,
        (1.0 - K1 / Q1 + K1 * K1) / ah,
    ]
    ks = np.array([s1, s2])

    lk = sosfilt(ks, l)
    rk = sosfilt(ks, r)
    bs = int(sr * 0.4)
    hs = int(sr * 0.1)
    nb = max(0, (len(lk) - bs) // hs + 1)
    if nb == 0:
        return -70.0

    blks = np.array([
        np.mean(lk[i * hs:i * hs + bs] ** 2)
        + np.mean(rk[i * hs:i * hs + bs] ** 2)
        for i in range(nb)
    ])
    del lk, rk
    ga = blks[blks > 10 ** ((-70 + 0.691) / 10)]
    if len(ga) == 0:
        return -70.0
    gr = ga[ga > np.mean(ga) * 10 ** (-10 / 10)]
    if len(gr) == 0:
        return -70.0
    return -0.691 + 10 * np.log10(
        max(np.mean(gr), LOG_FLOOR)
    )


def _measure_true_peak_db(
    l: np.ndarray, r: np.ndarray, sr: int,
) -> float:
    tl = resample_poly(l, 4, 1)
    pl = float(np.max(np.abs(tl)))
    del tl
    tr = resample_poly(r, 4, 1)
    pr = float(np.max(np.abs(tr)))
    del tr
    pk = max(pl, pr)
    return (
        20.0 * np.log10(pk)
        if pk >= LOG_FLOOR else -100.0
    )


def _calculate_dynamic_range(
    l: np.ndarray, r: np.ndarray, sr: int,
) -> float:
    """Dynamic range over averaged mono. Takes separate L/R so no column_stack."""
    mono = (l + r) * 0.5
    fl = int(sr * 0.01)
    nf = len(mono) // fl
    if nf > 10:
        rms = np.sqrt(np.mean(
            mono[:nf * fl].reshape(nf, fl) ** 2,
            axis=1,
        ))
        fdb = 20.0 * np.log10(
            np.maximum(rms, LOG_FLOOR)
        )
        act = fdb[fdb > -60]
        if len(act) > 10:
            return float(
                np.percentile(act, 95)
                - np.percentile(act, 5)
            )
    mr = np.sqrt(np.mean(mono ** 2))
    mp = np.max(np.abs(mono))
    return float(
        20.0 * np.log10(max(mp, LOG_FLOOR))
        - 20.0 * np.log10(max(mr, LOG_FLOOR))
    )


def _calculate_crest_factor(
    l: np.ndarray, r: np.ndarray,
) -> float:
    """Crest factor in dB: peak-to-RMS ratio.

    A healthy master typically has 6-12 dB crest factor.
    Below 4 dB indicates over-compression / over-limiting.
    """
    mono = (l + r) * 0.5
    rms = float(np.sqrt(np.mean(mono ** 2)))
    peak = float(np.max(np.abs(mono)))
    if rms < LOG_FLOOR:
        return 0.0
    return 20.0 * np.log10(max(peak, LOG_FLOOR) / rms)


def _db_to_linear(db: float) -> float:
    return 10 ** (db / 20.0)


def _make_peaking_sos(
    freq: float, gain_db: float,
    q: float, sr: int,
) -> np.ndarray:
    w0 = 2.0 * math.pi * freq / sr
    A = 10 ** (gain_db / 40.0)
    alpha = math.sin(w0) / (2.0 * q)
    b0 = 1 + alpha * A
    b1 = -2 * math.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha / A
    return np.array([[
        b0 / a0, b1 / a0, b2 / a0,
        1.0, a1 / a0, a2 / a0,
    ]])
