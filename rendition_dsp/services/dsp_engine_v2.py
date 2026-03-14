

"""
RENDITION_DSP Engine v2 — 14-Stage Mastering Chain

Signal flow:
  ① DC Remove → ② Gain Stage →
  ③ Transformer Sat → ④ Triode Tube → ⑤ Tape Emulation →
  ⑥ M/S Encode →
  ⑦ Parametric EQ → ⑧ Dynamic EQ → ⑨ 4-Band Comp →
  ⑩ Parallel Drive → ⑪ Freq-Dep Width → ⑫ Soft Clipper →
  M/S Decode →
  ⑬ TP Limiter v2 → ⑭ TPDF Dither

Self-correction convergence: binary search + linear refinement
LUFS measurement: ITU-R BS.1770-4 compliant (K-weight + double gating)
Saturation stages ③④⑤: 8x oversampling via resample_poly
"""

import io
import logging
import math
import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly, lfilter
from scipy.ndimage import maximum_filter1d

logger = logging.getLogger("rendition_dsp.dsp_v2")

# ──────────────────────────────────────────
# Constants
# ──────────────────────────────────────────
MAX_CONVERGENCE_LOOPS = 20
CONVERGENCE_TOLERANCE_DB = 0.1
LOG_FLOOR = 1e-10
# Max samples for full 8x OS; longer signals use chunked processing
_OS_CHUNK_SAMPLES = 2_000_000  # ~45s at 44.1kHz


# ══════════════════════════════════════════
# Public API (same signature as v1)
# ══════════════════════════════════════════
def master_audio(
    input_path: str,
    output_path: str,
    params: dict,
    target_lufs: float = -14.0,
    target_true_peak: float = -1.0,
) -> dict:
    """
    Apply 14-stage mastering chain with self-correction loop.
    Reads from input_path and writes to output_path.
    Returns metrics_dict.
    """
    # Decode
    data, sr = sf.read(input_path, dtype="float64")
    if data.ndim == 1:
        data = np.column_stack([data, data])

    left = data[:, 0].copy()
    right = data[:, 1].copy() if data.shape[1] >= 2 else data[:, 0].copy()

    # Measure before
    lufs_before = _calculate_lufs_bs1770(left, right, sr)
    peak_before = _measure_true_peak_db(left, right, sr)

    # ① DC Remove
    left = _remove_dc(left, sr)
    right = _remove_dc(right, sr)

    # Per-job dither seed (unique noise per mastering run)
    dither_seed = int.from_bytes(os.urandom(4), 'little')

    # ── Self-correction convergence loop (binary search + refinement) ──
    gain_adjustment = 0.0
    best_output = None
    convergence_loops = 0

    # Phase A: Binary search for coarse gain (6 iterations → ±0.375 dB)
    lo, hi = -12.0, 12.0
    for _ in range(6):
        convergence_loops += 1
        mid_gain = (lo + hi) / 2.0
        out_l, out_r = _apply_full_chain(left.copy(), right.copy(), sr, params, mid_gain, dither_seed)
        current_lufs = _calculate_lufs_bs1770(out_l, out_r, sr)
        if current_lufs < target_lufs:
            lo = mid_gain
        else:
            hi = mid_gain

    # Phase B: Linear refinement (±0.1 dB precision)
    gain_adjustment = (lo + hi) / 2.0
    for _ in range(MAX_CONVERGENCE_LOOPS):
        convergence_loops += 1
        out_l, out_r = _apply_full_chain(left.copy(), right.copy(), sr, params, gain_adjustment, dither_seed)
        current_lufs = _calculate_lufs_bs1770(out_l, out_r, sr)
        current_peak = _measure_true_peak_db(out_l, out_r, sr)

        lufs_diff = current_lufs - target_lufs
        peak_safe = current_peak <= target_true_peak + 0.1

        if abs(lufs_diff) < CONVERGENCE_TOLERANCE_DB and peak_safe:
            best_output = np.column_stack([out_l, out_r])
            logger.info(f"Converged in {convergence_loops} loops: LUFS={current_lufs:.2f}")
            break

        # Adaptive step
        if abs(lufs_diff) > 3:
            step = 0.5
        elif abs(lufs_diff) > 1:
            step = 0.2
        else:
            step = 0.1

        if lufs_diff > 0:
            gain_adjustment -= step
        else:
            gain_adjustment += step

        best_output = np.column_stack([out_l, out_r])

    if best_output is None:
        out_l, out_r = _apply_full_chain(left.copy(), right.copy(), sr, params, gain_adjustment, dither_seed)
        best_output = np.column_stack([out_l, out_r])

    # Final safety (true peak via 4x oversampling, not sample peak)
    ceiling = _db_to_linear(target_true_peak)
    tp_l = resample_poly(best_output[:, 0], 4, 1)
    tp_r = resample_poly(best_output[:, 1], 4, 1)
    true_peak = max(np.max(np.abs(tp_l)), np.max(np.abs(tp_r)))
    if true_peak > ceiling:
        best_output *= ceiling / true_peak
    best_output = np.clip(best_output, -ceiling, ceiling)

    # Measure after
    lufs_after = _calculate_lufs_bs1770(best_output[:, 0], best_output[:, 1], sr)
    peak_after = _measure_true_peak_db(best_output[:, 0], best_output[:, 1], sr)
    dr_after = _calculate_dynamic_range(best_output, sr)

    # Encode to 24-bit WAV directly to disk
    sf.write(output_path, best_output.astype(np.float32), sr, format="WAV", subtype="PCM_24")

    metrics = {
        "lufs_before": round(lufs_before, 1),
        "lufs_after": round(lufs_after, 1),
        "true_peak_before": round(peak_before, 1),
        "true_peak_after": round(peak_after, 1),
        "dynamic_range_after": round(dr_after, 1),
        "convergence_loops": convergence_loops,
        "gain_adjustment_db": round(gain_adjustment, 2),
        "target_lufs": target_lufs,
        "target_true_peak": target_true_peak,
        "engine_version": "v2_14stage",
    }

    return metrics


# ══════════════════════════════════════════
# Full Chain (stages ③–⑭)
# ══════════════════════════════════════════
def _apply_full_chain(
    left: np.ndarray,
    right: np.ndarray,
    sr: int,
    params: dict,
    gain_adj: float,
    dither_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply stages ③–⑭ to L/R channels, encoding M/S where appropriate."""

    # ② Gain Stage
    gain_linear = _db_to_linear(params.get("input_gain_db", 0) + gain_adj)
    left *= gain_linear
    right *= gain_linear

    # ③ Transformer Saturation (odd harmonics, 8x OS)
    left = _apply_transformer(left, sr, params)
    right = _apply_transformer(right, sr, params)

    # ④ Triode Tube (Koren model, 8x OS)
    left = _apply_triode(left, sr, params)
    right = _apply_triode(right, sr, params)

    # ⑤ Tape Emulation (compression + head bump + HF rolloff, 8x OS)
    left = _apply_tape(left, sr, params)
    right = _apply_tape(right, sr, params)

    # ⑥ M/S Encode (after saturation, before EQs/Comps)
    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    # ⑦ Parametric EQ (AI-controlled 4-band)
    mid = _apply_parametric_eq(mid, sr, params, "mid")
    side = _apply_parametric_eq(side, sr, params, "side")

    # ⑧ Dynamic EQ (frequency-selective compression)
    mid = _apply_dynamic_eq(mid, sr, params)
    side = _apply_dynamic_eq(side, sr, params)

    # ⑨ 4-Band Multiband Compressor
    mid = _apply_multiband_comp(mid, sr, params)
    side = _apply_multiband_comp(side, sr, params)

    # ⑩ Parallel Drive (tanh saturation + HPF + air shelf)
    parallel_wet = params.get("parallel_wet", 0.18)
    mid = _neuro_drive(mid, sr, wet=parallel_wet)
    side = _neuro_drive(side, sr, wet=parallel_wet)

    # ⑪ Frequency-Dependent Stereo Width (operates on M/S pair)
    mid, side = _apply_freq_dep_width(mid, side, sr, params)

    # ⑫ Soft Clipper
    mid = _soft_clipper(mid, threshold=0.98)
    side = _soft_clipper(side, threshold=0.98)

    # M/S Decode
    out_left = mid + side
    out_right = mid - side

    # ⑬ True Peak Limiter v2 (8x OS + Lookahead, stereo-linked)
    ceil_db = params.get("limiter_ceil_db", -0.1)
    out_left, out_right = _apply_true_peak_limiter_stereo(out_left, out_right, sr, ceil_db)

    # ⑭ TPDF Dither
    out_left = _apply_dither(out_left, target_bits=24, channel_idx=0, seed=dither_seed)
    out_right = _apply_dither(out_right, target_bits=24, channel_idx=1, seed=dither_seed)

    return out_left, out_right


# ══════════════════════════════════════════
# ① DC Offset Removal
# ══════════════════════════════════════════
def _remove_dc(buf: np.ndarray, sr: int) -> np.ndarray:
    """Remove DC offset with 1st-order IIR HPF at 5 Hz."""
    fc = 5.0
    w0 = 2.0 * np.pi * fc / sr
    # 1st-order HPF: H(z) = (1 - z^-1) / (1 - (1-w0)*z^-1)
    b = np.array([1.0, -1.0])
    a = np.array([1.0, -(1.0 - w0)])
    # Normalize
    b = b / (1.0 + w0)
    a[1] = a[1] / (1.0 + w0)
    a[0] = 1.0
    return lfilter(b, a, buf)


# ══════════════════════════════════════════
# ③ Transformer Saturation (tanh waveshaper)
# ══════════════════════════════════════════
def _apply_transformer(buf: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """Transformer saturation — tanh waveshaper with IIR smoothing (odd harmonics)."""
    saturation = params.get("transformer_saturation", 0.3)
    mix = params.get("transformer_mix", 0.4)

    if saturation < 0.01 and mix < 0.01:
        return buf

    # 8x oversample
    up = resample_poly(buf, 8, 1)

    # tanh waveshaper: B = tanh(H/3)
    drive = saturation * 5.0 + 0.5
    H = up * drive
    B = np.tanh(H / 3.0)

    # IIR smoothing: y[n] = (1-m)*B[n] + m*y[n-1]
    mem_coeff = 0.15 * saturation
    if mem_coeff > 0.001:
        b_hyst = np.array([1.0 - mem_coeff])
        a_hyst = np.array([1.0, -mem_coeff])
        B = lfilter(b_hyst, a_hyst, B)

    # Wet/dry blend
    wet = saturation * 0.6
    result = up * (1.0 - wet) + B * wet

    # HF rolloff (gentle anti-aliasing)
    rolloff_factor = 0.02
    fc_loss = 18000 * (1.0 - rolloff_factor * 10)
    if fc_loss < sr * 4 * 0.49:  # Below Nyquist of oversampled rate
        sos_lp = butter(1, fc_loss, btype='low', fs=sr * 8, output='sos')
        result = sosfilt(sos_lp, result)

    # Downsample
    return resample_poly(result, 1, 8)[:len(buf)]


# ══════════════════════════════════════════
# ④ Triode Tube (Koren Model)
# ══════════════════════════════════════════
def _apply_triode(buf: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """Triode model (Koren transfer function with bias control)."""
    drive = params.get("triode_drive", 0.4)
    bias = params.get("triode_bias", -1.2)
    mix = params.get("triode_mix", 0.5)

    if drive < 0.01 and mix < 0.01:
        return buf

    # 8x oversample
    up = resample_poly(buf, 8, 1)

    # Koren triode model parameters
    Kp = 600.0
    Kvb = 300.0
    Ex = 1.4
    Vp = 250.0  # Plate voltage
    mu = 100.0   # Amplification factor

    # Grid voltage
    input_gain = drive * 8.0 + 0.5
    Vg = up * input_gain + bias

    # Koren equation: E1 = (Vp/Kp) * log(1 + exp(Kp * (1/mu + Vg/sqrt(Kvb + Vp^2))))
    sqrt_term = math.sqrt(Kvb + Vp * Vp)
    inner = Kp * (1.0 / mu + Vg / sqrt_term)
    # Numerically stable log(1+exp(x))
    E1 = (Vp / Kp) * np.where(inner > 20.0, inner, np.log1p(np.exp(np.clip(inner, -20, 20))))

    # Plate current: Ip = E1^Ex
    Ip = np.power(np.maximum(E1, 0.0), Ex)

    # Grid current (positive grid voltage clipping)
    Kg = 1060.0
    Ig = np.where(Vg > 0, np.expm1(np.clip(Vg / Kg, -20, 20)), 0.0)
    Ip = Ip + Ig * 0.1

    # Normalize to [-1, +1]
    max_Ip = np.max(np.abs(Ip)) + 1e-10
    saturated = Ip / max_Ip

    # DC block (coupling capacitor) via HPF at 10 Hz
    fc_dc = 10.0
    w_dc = 2.0 * np.pi * fc_dc / (sr * 8)
    b_dc = np.array([1.0, -1.0]) / (1.0 + w_dc)
    a_dc = np.array([1.0, -(1.0 - w_dc) / (1.0 + w_dc)])
    saturated = lfilter(b_dc, a_dc, saturated)

    # Wet/dry
    result = up * (1.0 - mix) + saturated * mix

    # Downsample
    return resample_poly(result, 1, 8)[:len(buf)]


# ══════════════════════════════════════════
# ⑤ Tape Emulation
# ══════════════════════════════════════════
def _apply_tape(buf: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """Tape emulation (arctan saturation + head bump + HF rolloff).

    Note: tape_speed only affects HF rolloff cutoff frequency.
    Wow/flutter is not implemented.
    """
    saturation = params.get("tape_saturation", 0.3)
    mix = params.get("tape_mix", 0.4)

    if saturation < 0.01 and mix < 0.01:
        return buf

    speed_ips = params.get("tape_speed", 30.0)
    bump_freq = 80.0
    bump_gain_db = 2.0 * saturation

    # 8x oversample
    up = resample_poly(buf, 8, 1)
    sr_up = sr * 8

    # Tape compression: arctan saturation
    drive = saturation * 3.0 + 0.5
    arctan_norm = np.arctan(drive)  # normalization factor
    compressed = np.arctan(up * drive) / arctan_norm

    # Head bump: peaking EQ at bump_freq
    if bump_gain_db > 0.1:
        Q = 0.8
        w0 = 2.0 * np.pi * bump_freq / sr_up
        alpha = math.sin(w0) / (2.0 * Q)
        A = 10 ** (bump_gain_db / 40.0)
        # Peaking EQ biquad coefficients → SOS
        b0 = 1 + alpha * A
        b1 = -2 * math.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * math.cos(w0)
        a2 = 1 - alpha / A
        sos_bump = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
        compressed = sosfilt(sos_bump, compressed)

    # HF rolloff (tape speed dependent)
    hf_cutoff = 12000 + speed_ips * 350
    if hf_cutoff < sr_up * 0.49:
        sos_hf = butter(2, hf_cutoff, btype='low', fs=sr_up, output='sos')
        compressed = sosfilt(sos_hf, compressed)

    # Wet/dry
    result = up * (1.0 - mix) + compressed * mix

    # Downsample
    return resample_poly(result, 1, 8)[:len(buf)]


# ══════════════════════════════════════════
# ⑧ Dynamic EQ
# ══════════════════════════════════════════
def _apply_dynamic_eq(buf: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """Frequency-selective dynamic processing."""
    if not params.get("dyn_eq_enabled", True):
        return buf

    # Default bands: Low Tame, Mud Cut, De-Harsh, Air Boost
    default_bands = [
        {"freq": 80, "q": 0.8, "threshold_db": -12, "max_gain_db": -3,
         "attack_ms": 20.0, "release_ms": 150.0},
        {"freq": 300, "q": 1.0, "threshold_db": -15, "max_gain_db": -2,
         "attack_ms": 15.0, "release_ms": 120.0},
        {"freq": 5000, "q": 1.2, "threshold_db": -18, "max_gain_db": -4,
         "attack_ms": 10.0, "release_ms": 80.0},
        {"freq": 12000, "q": 0.8, "threshold_db": -25, "max_gain_db": 2,
         "attack_ms": 5.0, "release_ms": 60.0},
    ]
    bands = params.get("dyn_eq_bands", default_bands)
    result = buf.copy()

    for band in bands:
        freq = band["freq"]
        q = band["q"]
        threshold_db = band["threshold_db"]
        max_gain_db = band["max_gain_db"]
        attack_ms = band.get("attack_ms", 10.0)
        release_ms = band.get("release_ms", 80.0)

        if freq >= sr * 0.49 or freq < 20:
            continue

        # Sidechain: BPF to extract target band level
        bw = freq / q
        low = max(freq - bw / 2, 20)
        high = min(freq + bw / 2, sr * 0.49 - 1)
        if low >= high:
            continue
        sos_bp = butter(2, [low, high], btype='band', fs=sr, output='sos')
        sidechain = sosfilt(sos_bp, buf)

        # Dual-envelope follower (fast attack + slow release, vectorized)
        abs_sc = np.abs(sidechain)
        attack_coeff = math.exp(-1.0 / max(1, attack_ms / 1000.0 * sr))
        release_coeff = math.exp(-1.0 / max(1, release_ms / 1000.0 * sr))

        b_fast = np.array([1.0 - attack_coeff])
        a_fast = np.array([1.0, -attack_coeff])
        env_fast = lfilter(b_fast, a_fast, abs_sc)

        b_slow = np.array([1.0 - release_coeff])
        a_slow = np.array([1.0, -release_coeff])
        env_slow = lfilter(b_slow, a_slow, abs_sc)

        envelope = np.maximum(env_fast, env_slow)

        # Dynamic gain
        threshold_lin = _db_to_linear(threshold_db)
        max_gain_lin = _db_to_linear(max_gain_db)

        gain = np.where(
            envelope > threshold_lin,
            1.0 + (max_gain_lin - 1.0) * np.minimum(
                (envelope - threshold_lin) / (threshold_lin * 2.0 + 1e-10), 1.0
            ),
            1.0,
        )

        # Apply: add filtered band weighted by dynamic gain
        eq_signal = sosfilt(sos_bp, result)
        result = result + eq_signal * (gain - 1.0)

    return result


# ══════════════════════════════════════════
# ⑨ 4-Band Multiband Compressor
# ══════════════════════════════════════════
def _apply_multiband_comp(buf: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """4-band multiband compressor with dual-envelope dynamics."""
    threshold_db = params.get("comp_threshold_db", -12)
    ratio = params.get("comp_ratio", 2.5)
    base_attack = params.get("comp_attack_sec", 0.01)
    base_release = params.get("comp_release_sec", 0.15)

    # Crossover frequencies
    xovers = [80, 300, 4000]

    # Split into 4 bands using Butterworth 4th order filters
    bands = _split_4bands(buf, sr, xovers)
    processed = []

    for i, band_signal in enumerate(bands):
        # Per-band threshold adjustment
        band_thresh = threshold_db + (i - 1) * 2  # Higher bands slightly different threshold
        band_thresh = max(band_thresh, -30)

        compressed = _compress_band(
            band_signal, sr, band_thresh, ratio, base_attack, base_release
        )
        processed.append(compressed)

    # Sum bands
    return sum(processed)


def _split_4bands(buf: np.ndarray, sr: int, xovers: list) -> list:
    """Split signal into 4 bands using Linkwitz-Riley 4th order (LR4) crossovers.

    LR4 = two cascaded Butterworth 2nd-order filters.  LP + HP sum to unity
    (complementary), eliminating the ±3 dB ripple of naive Butterworth splits.
    """
    bands = []
    remaining = buf.copy()

    for xover in xovers:
        if xover >= sr * 0.49:
            bands.append(np.zeros_like(buf))
            continue
        # LR4 = Butterworth 2nd-order applied twice (cascaded)
        sos_lp2 = butter(2, xover, btype='low', fs=sr, output='sos')
        sos_hp2 = butter(2, xover, btype='high', fs=sr, output='sos')
        # Cascade: apply twice for LR4
        low = sosfilt(sos_lp2, sosfilt(sos_lp2, remaining))
        remaining = sosfilt(sos_hp2, sosfilt(sos_hp2, remaining))
        bands.append(low)

    bands.append(remaining)  # Highest band
    return bands


def _compress_band(
    buf: np.ndarray, sr: int,
    threshold_db: float, ratio: float,
    base_attack: float, base_release: float,
) -> np.ndarray:
    """Single-band dual-envelope compression."""
    threshold_lin = _db_to_linear(threshold_db)

    # Envelope follower (smoothed via lfilter)
    abs_signal = np.abs(buf)

    # Dual-envelope: fast (attack) + slow (release) smoothing
    attack_coeff = math.exp(-1.0 / max(1, base_attack * sr))
    release_coeff = math.exp(-1.0 / max(1, base_release * sr))

    # Fast envelope (attack-dominant)
    b_fast = np.array([1.0 - attack_coeff])
    a_fast = np.array([1.0, -attack_coeff])
    env_fast = lfilter(b_fast, a_fast, abs_signal)

    # Slow envelope (release-dominant)
    b_slow = np.array([1.0 - release_coeff])
    a_slow = np.array([1.0, -release_coeff])
    env_slow = lfilter(b_slow, a_slow, abs_signal)

    # Use max of fast/slow envelopes
    envelope = np.maximum(env_fast, env_slow)

    # Gain reduction
    gain = np.ones_like(buf)
    over_mask = envelope > threshold_lin
    if np.any(over_mask):
        over_db = 20.0 * np.log10(np.maximum(envelope[over_mask] / threshold_lin, LOG_FLOOR))
        gain_reduction_db = over_db * (1.0 - 1.0 / ratio)
        gain[over_mask] = _db_to_linear(-gain_reduction_db)

    # Smoothing (anti-zipper, 2ms)
    smooth_samples = max(int(sr * 0.002), 1)
    kernel = np.ones(smooth_samples) / smooth_samples
    gain = np.convolve(gain, kernel, mode='same')

    return buf * gain


# ══════════════════════════════════════════
# ⑦ Parametric EQ
# ══════════════════════════════════════════
def _apply_parametric_eq(buf: np.ndarray, sr: int, params: dict, ch_type: str) -> np.ndarray:
    """4-band parametric EQ using biquad filters in SOS form."""
    # Build EQ bands from params
    eq_config = [
        {"type": "lowshelf", "freq": 80, "gain_db": params.get("eq_low_shelf_gain_db", 0)},
        {"type": "peaking", "freq": 400, "q": 1.0, "gain_db": params.get("eq_low_mid_gain_db", 0)},
        {"type": "peaking", "freq": 3000, "q": 1.0, "gain_db": params.get("eq_high_mid_gain_db", 0)},
        {"type": "highshelf", "freq": 8000, "gain_db": params.get("eq_high_shelf_gain_db", 0)},
    ]

    # M/S specific adjustments
    if ch_type == "side":
        side_high = params.get("ms_side_high_gain_db", 0)
        eq_config.append(
            {"type": "highshelf", "freq": 4000, "gain_db": side_high * 0.5}
        )
    else:
        mid_low = params.get("ms_mid_low_gain_db", 0)
        eq_config.append(
            {"type": "lowshelf", "freq": 200, "gain_db": mid_low * 0.5}
        )

    result = buf.copy()
    for band in eq_config:
        gain_db = band["gain_db"]
        if abs(gain_db) < 0.05:
            continue

        freq = float(band["freq"])
        if freq >= sr * 0.49 or freq < 10:
            continue

        sos = _make_eq_sos(band, sr)
        if sos is not None:
            result = sosfilt(sos, result)

    return result


def _make_eq_sos(band: dict, sr: int) -> np.ndarray | None:
    """Build SOS biquad coefficients for a single EQ band."""
    freq = band["freq"]
    gain_db = band["gain_db"]
    eq_type = band["type"]
    Q = band.get("q", 0.707)

    w0 = 2.0 * math.pi * freq / sr
    A = 10 ** (gain_db / 40.0)
    alpha = math.sin(w0) / (2.0 * Q)

    if eq_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * math.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * math.cos(w0)
        a2 = 1 - alpha / A
    elif eq_type == "lowshelf":
        sq = 2.0 * math.sqrt(A) * alpha
        b0 = A * ((A + 1) - (A - 1) * math.cos(w0) + sq)
        b1 = 2 * A * ((A - 1) - (A + 1) * math.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * math.cos(w0) - sq)
        a0 = (A + 1) + (A - 1) * math.cos(w0) + sq
        a1 = -2 * ((A - 1) + (A + 1) * math.cos(w0))
        a2 = (A + 1) + (A - 1) * math.cos(w0) - sq
    elif eq_type == "highshelf":
        sq = 2.0 * math.sqrt(A) * alpha
        b0 = A * ((A + 1) + (A - 1) * math.cos(w0) + sq)
        b1 = -2 * A * ((A - 1) + (A + 1) * math.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * math.cos(w0) - sq)
        a0 = (A + 1) - (A - 1) * math.cos(w0) + sq
        a1 = 2 * ((A - 1) - (A + 1) * math.cos(w0))
        a2 = (A + 1) - (A - 1) * math.cos(w0) - sq
    else:
        return None

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


# ══════════════════════════════════════════
# ⑪ Frequency-Dependent Stereo Width
# ══════════════════════════════════════════
def _apply_freq_dep_width(
    mid: np.ndarray, side: np.ndarray, sr: int, params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Low-mono / high-wide stereo processing with mono compatibility check."""
    low_mono_freq = 200.0
    high_wide_freq = 4000.0
    low_mono_amount = params.get("stereo_low_mono", 0.8)
    high_wide_amount = params.get("stereo_high_wide", 1.15)
    global_width = params.get("stereo_width", 1.0)

    # Split side into 3 frequency bands
    if low_mono_freq >= sr * 0.49 or high_wide_freq >= sr * 0.49:
        return mid, side * global_width

    sos_lp = butter(4, low_mono_freq, btype='low', fs=sr, output='sos')
    sos_hp = butter(4, high_wide_freq, btype='high', fs=sr, output='sos')
    sos_bp = butter(4, [low_mono_freq, high_wide_freq], btype='band', fs=sr, output='sos')

    side_low = sosfilt(sos_lp, side)
    side_mid_band = sosfilt(sos_bp, side)
    side_high = sosfilt(sos_hp, side)

    # Apply per-band width
    side_low *= (1.0 - low_mono_amount)  # Reduce low-end side → mono
    side_mid_band *= global_width
    side_high *= high_wide_amount        # Boost high-end side → wider

    side_out = side_low + side_mid_band + side_high

    # Mono compatibility safety check
    mid_energy = np.sum(mid ** 2) + 1e-10
    side_energy = np.sum(side_out ** 2)
    if side_energy > mid_energy * 0.5:
        safety_gain = math.sqrt(mid_energy * 0.45 / max(side_energy, 1e-10))
        side_out *= safety_gain

    return mid, side_out


# ══════════════════════════════════════════
# ⑫ Soft Clipper
# ══════════════════════════════════════════
def _soft_clipper(signal: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    """Transient-protective soft clipper with tanh shaping."""
    slope = 0.04
    output = signal.copy()
    mask = np.abs(signal) > threshold
    if not np.any(mask):
        return output
    over = np.abs(signal[mask]) - threshold
    clipped = threshold + slope * np.tanh(over / slope)
    output[mask] = np.sign(signal[mask]) * clipped
    return output


# ══════════════════════════════════════════
# ⑬ True Peak Limiter v2
# ══════════════════════════════════════════
def _apply_true_peak_limiter_stereo(
    left: np.ndarray, right: np.ndarray, sr: int, ceiling_db: float = -0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Stereo-linked True Peak limiter with 8x OS detection + lookahead.

    Computes gain reduction from the max true-peak across both channels,
    then applies the same gain curve to both.  This preserves the stereo
    image (no independent L/R pumping).
    """
    ceiling = _db_to_linear(ceiling_db)
    lookahead_ms = 5.0
    release_ms = 50.0
    lookahead_samples = max(1, int(lookahead_ms / 1000.0 * sr))
    n = len(left)

    # 1. Detect true peak via 8x oversampling — BOTH channels
    up_l = resample_poly(left, 8, 1)
    up_r = resample_poly(right, 8, 1)
    # Stereo-linked: take max of both channels per sample
    abs_up = np.maximum(np.abs(up_l), np.abs(up_r))

    # 2. Map oversampled peaks back to original sample rate
    usable = n * 8
    if usable > len(abs_up):
        usable = (len(abs_up) // 8) * 8
        n_full = usable // 8
    else:
        n_full = n
    peak_per_sample = abs_up[:n_full * 8].reshape(n_full, 8).max(axis=1)
    if n_full < n:
        remainder_peak = np.max(abs_up[n_full * 8:]) if n_full * 8 < len(abs_up) else 0.0
        peak_per_sample = np.append(peak_per_sample, np.full(n - n_full, remainder_peak))

    # 3. Lookahead: forward-looking rolling max
    if lookahead_samples > 1:
        peak_ahead = _rolling_max(peak_per_sample, lookahead_samples)
    else:
        peak_ahead = peak_per_sample.copy()

    # 4. Gain computation (single stereo-linked curve)
    gain = np.where(peak_ahead > ceiling, ceiling / (peak_ahead + 1e-10), 1.0)

    # 5. Release smoothing (instant attack, fixed release)
    release_coeff = math.exp(-1.0 / max(1, release_ms / 1000.0 * sr))
    b_rel = np.array([1.0 - release_coeff])
    a_rel = np.array([1.0, -release_coeff])
    smoothed = gain.copy()
    smoothed = np.minimum(gain, lfilter(b_rel, a_rel, smoothed))
    smoothed = np.minimum(gain, lfilter(b_rel, a_rel, smoothed))
    smoothed = np.minimum(gain, lfilter(b_rel, a_rel, smoothed))

    # 6. Apply same gain to both channels (stereo-linked)
    return left * smoothed, right * smoothed


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling maximum over a forward-looking window (O(N) vectorized)."""
    if window <= 1:
        return np.copy(arr)
    # Center is shifted by -(window // 2) to look forward
    return maximum_filter1d(arr, size=window, origin=-(window // 2), mode='constant', cval=0.0)


# ══════════════════════════════════════════
# ⑩ Parallel Drive (Enhanced Neuro-Drive)
# ══════════════════════════════════════════
def _neuro_drive(buf: np.ndarray, sr: int, wet: float = 0.18) -> np.ndarray:
    """Parallel saturation (tanh waveshaper) + HPF + high shelf for energy."""
    if wet < 0.01:
        return buf

    # tanh saturation (not compression — no threshold/ratio/envelope)
    compressed = np.tanh(buf * 3.0) * 0.5

    # HPF at 80 Hz on compressed signal (remove mud)
    if sr > 200:
        sos_hpf = butter(2, 80, btype='high', fs=sr, output='sos')
        compressed = sosfilt(sos_hpf, compressed)

    # High shelf boost at 8kHz for air
    sos_air = _make_eq_sos({"type": "highshelf", "freq": 8000, "gain_db": 2.0, "q": 0.707}, sr)
    if sos_air is not None and 8000 < sr * 0.49:
        compressed = sosfilt(sos_air, compressed)

    return buf * (1.0 - wet) + compressed * wet


# ══════════════════════════════════════════
# ⑭ TPDF Dither
# ══════════════════════════════════════════
def _apply_dither(buf: np.ndarray, target_bits: int = 24, channel_idx: int = 0, seed: int = 0) -> np.ndarray:
    """Flat TPDF dither (no noise shaping).

    TPDF = sum of two independent uniform random variables → triangular PDF.
    This is the standard non-shaped dither used in professional mastering.
    No HP pre-filter is applied; the noise spectrum is flat.
    """
    levels = 2 ** (target_bits - 1)
    lsb = 1.0 / levels

    # Per-job + per-channel TPDF noise (unique per mastering run)
    rng = np.random.RandomState(seed + channel_idx)
    r1 = rng.uniform(-0.5, 0.5, len(buf))
    r2 = rng.uniform(-0.5, 0.5, len(buf))
    tpdf_noise = (r1 + r2) * lsb

    # Vectorized quantization (flat TPDF — no shaping filter)
    dithered = buf + tpdf_noise
    quantized = np.round(dithered * levels) / levels

    return quantized


# ══════════════════════════════════════════
# BS.1770-4 LUFS Measurement
# ══════════════════════════════════════════
def _calculate_lufs_bs1770(left: np.ndarray, right: np.ndarray, sr: int) -> float:
    """EBU R128 / ITU-R BS.1770-4 compliant Integrated Loudness."""
    if len(left) < 1024:
        return -70.0

    # Stage 1: K-weighting filter (pre-filter + RLB)
    k_sos = _build_k_weight_sos(sr)
    left_k = sosfilt(k_sos, left)
    right_k = sosfilt(k_sos, right)

    # Stage 2: 400ms blocks with 75% overlap
    block_size = int(sr * 0.4)
    hop_size = int(sr * 0.1)
    if block_size < 1:
        return -70.0

    n_blocks = max(0, (len(left_k) - block_size) // hop_size + 1)
    if n_blocks == 0:
        return -70.0

    # Vectorized block power computation
    blocks = np.zeros(n_blocks)
    for i in range(n_blocks):
        start = i * hop_size
        end = start + block_size
        l_block = left_k[start:end]
        r_block = right_k[start:end]
        blocks[i] = np.mean(l_block ** 2) + np.mean(r_block ** 2)

    # Stage 3: Absolute gating (-70 LUFS)
    abs_threshold = 10 ** ((-70 + 0.691) / 10)
    gated_abs = blocks[blocks > abs_threshold]
    if len(gated_abs) == 0:
        return -70.0

    # Stage 4: Relative gating (-10 LU below absolute-gated mean)
    abs_mean = np.mean(gated_abs)
    rel_threshold = abs_mean * 10 ** (-10 / 10)
    gated_rel = gated_abs[gated_abs > rel_threshold]
    if len(gated_rel) == 0:
        return -70.0

    mean_power = np.mean(gated_rel)
    return -0.691 + 10 * np.log10(max(mean_power, LOG_FLOOR))


def _build_k_weight_sos(sr: int) -> np.ndarray:
    """Build K-weighting filter as SOS (2 biquad stages).

    Stage 1: Pre-filter (high shelf, +4 dB, ~1500 Hz)
    Stage 2: RLB weighting (HPF, ~38 Hz, 2nd order)
    """
    # Stage 1: High shelf at 1681 Hz, +4 dB (ITU-R BS.1770 specification)
    f0 = 1681.974450955533
    Q = 0.7071752369554196
    K = math.tan(math.pi * f0 / sr)
    Vh = 10 ** (4.0 / 20.0)  # +4 dB = 1.5849
    Vb = Vh ** 0.4996
    a0_hs = 1.0 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / a0_hs
    b1 = 2.0 * (K * K - Vh) / a0_hs
    b2 = (Vh - Vb * K / Q + K * K) / a0_hs
    a1 = 2.0 * (K * K - 1.0) / a0_hs
    a2 = (1.0 - K / Q + K * K) / a0_hs

    stage1 = [b0, b1, b2, 1.0, a1, a2]

    # Stage 2: High-pass at 38.13547087602444 Hz (RLB weighting)
    f1 = 38.13547087602444
    Q1 = 0.5003270373238773
    K1 = math.tan(math.pi * f1 / sr)
    a0_hp = 1.0 + K1 / Q1 + K1 * K1
    b0_hp = 1.0 / a0_hp
    b1_hp = -2.0 / a0_hp
    b2_hp = 1.0 / a0_hp
    a1_hp = 2.0 * (K1 * K1 - 1.0) / a0_hp
    a2_hp = (1.0 - K1 / Q1 + K1 * K1) / a0_hp

    stage2 = [b0_hp, b1_hp, b2_hp, 1.0, a1_hp, a2_hp]

    return np.array([stage1, stage2])


def _measure_true_peak_db(left: np.ndarray, right: np.ndarray, sr: int) -> float:
    """Measure True Peak (dBTP) via 4x oversampling with sinc interpolation."""
    # resample_poly uses polyphase FIR (sinc-like) by default
    left_up = resample_poly(left, 4, 1)
    right_up = resample_poly(right, 4, 1)
    peak = max(np.max(np.abs(left_up)), np.max(np.abs(right_up)))
    if peak < LOG_FLOOR:
        return -100.0
    return 20.0 * np.log10(peak)


def _calculate_dynamic_range(stereo: np.ndarray, sr: int) -> float:
    """Percentile-based dynamic range (95th - 5th of short-term RMS in dB).

    Matches the algorithm in audio_analysis._analyze_dynamics.
    """
    mono = np.mean(stereo, axis=1) if stereo.ndim == 2 else stereo
    frame_len = int(sr * 0.01)  # 10ms frames
    n_frames = len(mono) // frame_len
    if n_frames > 10:
        frames = mono[:n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        frame_db = 20.0 * np.log10(np.maximum(frame_rms, LOG_FLOOR))
        active = frame_db[frame_db > -60]
        if len(active) > 10:
            return float(np.percentile(active, 95) - np.percentile(active, 5))
    # Fallback: peak-to-RMS
    peak = np.max(np.abs(mono))
    rms = np.sqrt(np.mean(mono ** 2))
    return 20.0 * np.log10(max(peak, LOG_FLOOR)) - 20.0 * np.log10(max(rms, LOG_FLOOR))


# ══════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════
def _db_to_linear(db: float) -> float:
    return 10 ** (db / 20.0)
