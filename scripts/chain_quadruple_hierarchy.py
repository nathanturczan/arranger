#!/usr/bin/env python3
"""
Chain Feldman 3voices samples using proper Quadruple Hierarchy logic.

Algorithm:
1. Start with random sample
2. Take last_collection (pitch classes at end)
3. Infer chord supersets that contain last_collection (using chords_no_supersets.json)
4. Pick a chord from candidates
5. Find another sample whose first_collection ⊆ that chord (with transposition)
6. Repeat using new sample's last_collection
7. Continue until samples exhausted (max 2 uses per sample)

Samples with semitone clusters (3+ consecutive semitones) are disqualified.

This implements Issue #26: Quadruple Hierarchy Chaining.
"""

import bisect
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from pydub import AudioSegment
import pretty_midi
import numpy as np
import sys

# Add src to path for arranger package imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arranger.samples.registry import (
    # Primary sample libraries
    SAMPLES_DIR, AUDIO_DIR, MANIFEST_PATH,
    HANDEL_DIR, HANDEL_AUDIO_DIR, HANDEL_MANIFEST_PATH,
    # Continuous layers
    BASSFLUTE_DIR, BASSFLUTE_AUDIO_DIR, BASSFLUTE_MANIFEST_PATH,
    AVERYVIOLIN_DIR, AVERYVIOLIN_AUDIO_DIR, AVERYVIOLIN_MANIFEST_PATH,
    DICTAMEL_DIR, DICTAMEL_AUDIO_DIR, DICTAMEL_MANIFEST_PATH,
    TRICHORDS_DIR, TRICHORDS_AUDIO_DIR, TRICHORDS_MANIFEST_PATH,
    SCELSIPEZZI_DIR, SCELSIPEZZI_AUDIO_DIR, SCELSIPEZZI_MANIFEST_PATH,
    # Interval layers
    ORGANETTA_DIR, ORGANETTA_AUDIO_DIR, ORGANETTA_MANIFEST_PATH,
    FEEDBACK_DIR, FEEDBACK_AUDIO_DIR, FEEDBACK_MANIFEST_PATH,
    STYLO_DIR, STYLO_AUDIO_DIR, STYLO_MANIFEST_PATH,
    TREMOLO_OCT_DIR, TREMOLO_OCT_AUDIO_DIR, TREMOLO_OCT_MANIFEST_PATH,
    # Cloud layers
    BRODERO_DIR, BRODERO_MANIFEST_PATH,
    JICELLO_DIR, JICELLO_AUDIO_DIR, JICELLO_MANIFEST_PATH,
    GOTHICHARP_DIR, GOTHICHARP_AUDIO_DIR, GOTHICHARP_MANIFEST_PATH,
    GENTLEHARPSI_DIR, GENTLEHARPSI_AUDIO_DIR, GENTLEHARPSI_MANIFEST_PATH,
    LAKEN_DIR, LAKEN_AUDIO_DIR, LAKEN_MANIFEST_PATH,
    # Rhythmic layers
    MINORCHORDBEAT_DIR, MINORCHORDBEAT_AUDIO_DIR, MINORCHORDBEAT_MANIFEST_PATH,
    MUTEBOWL_DIR, MUTEBOWL_AUDIO_DIR, MUTEBOWL_MANIFEST_PATH,
    # Chord/quality layers
    MINORCHORDS_DIR, MINORCHORDS_AUDIO_DIR, MINORCHORDS_MANIFEST_PATH,
    MAJORCHORDS_DIR, MAJORCHORDS_AUDIO_DIR, MAJORCHORDS_MANIFEST_PATH,
    # Single-note layers
    PROPHETFALSE_DIR, PROPHETFALSE_AUDIO_DIR, PROPHETFALSE_MANIFEST_PATH,
    HARMONICKER_DIR, HARMONICKER_AUDIO_DIR, HARMONICKER_MANIFEST_PATH,
    # Progression samples
    GLAZ_SAX_DIR, GLAZ_SAX_AUDIO_DIR, GLAZ_SAX_MANIFEST_PATH,
    HYACINTHE_DIR, HYACINTHE_AUDIO_DIR, HYACINTHE_MANIFEST_PATH,
    KRAUS_DIR, KRAUS_AUDIO_DIR, KRAUS_MANIFEST_PATH,
    # Chord-triggered layers
    GODETTE_DIR, GODETTE_AUDIO_DIR, GODETTE_MANIFEST_PATH,
    # Constants
    MIN_TRANSPOSITION, MAX_TRANSPOSITION,
    GLISSANDO_MS, GLISSANDO_ANTICIPATION_MS,
    SAMPLE_NORMALIZE_DB,
    OUTPUT_DIR, CHORDS_JSON_PATH,
    NOTE_NAMES,
)
from arranger.audio.normalize import apply_gain_db
from arranger.layers import LayerConfig, LayerType, SelectionMode, render_layer

# =============================================================================
# RENDER SWITCHES - Toggle layers for fast preview
# =============================================================================
# Core layers (your preferred setup: bf + bro + jic)
ENABLE_BASSFLUTE = True
ENABLE_BRODERO = True
ENABLE_JICELLO = True
ENABLE_ORGANETTA = False
ENABLE_GENTLEHARPSI = False

# Heavy/slow layers
ENABLE_FEEDBACK = False
ENABLE_CLOUDS = False  # Gothic harp, Laken
ENABLE_PROGRESSIONS = False  # GlazSax, Hyacinthe, Kraus

# MIDI synth layers
ENABLE_MIDI_SYNTH = True
ENABLE_SYNTH_BASS = True

# Preview mode: when True, uses the switches above
FAST_PREVIEW = False

# =============================================================================
# AUDIO SAMPLE RATE - All audio is resampled to this rate for consistent mixing
# =============================================================================
TARGET_SAMPLE_RATE = 44100

# =============================================================================

# Global chord dictionary (loaded once)
CHORD_DICTIONARY: Dict[str, Dict] = {}

# Global bass flute samples (loaded once)
BASSFLUTE_SAMPLES: Dict[str, Dict] = {}

# Global Brodero samples (loaded once)
BRODERO_SAMPLES: Dict[str, Dict] = {}

# Global Jicello samples (loaded once)
JICELLO_SAMPLES: List[Dict] = []

# Global Organetta samples (loaded once)
ORGANETTA_SAMPLES: Dict[str, Dict] = {}

# Global MinorChordBeat samples (loaded once)
MINORCHORDBEAT_SAMPLES: Dict[str, Dict] = {}

# Global MuteBowl samples (loaded once)
MUTEBOWL_SAMPLES: Dict[str, Dict] = {}

# Global Major/Minor chord samples (loaded once)
MINORCHORDS_SAMPLES: List[Dict] = []
MAJORCHORDS_SAMPLES: List[Dict] = []

# Global Prophet False samples (loaded once)
PROPHETFALSE_SAMPLES: List[Dict] = []

# Global Harmonicker samples (loaded once)
HARMONICKER_SAMPLES: List[Dict] = []

# Global Gothic Harp samples (loaded once)
GOTHICHARP_SAMPLES: List[Dict] = []

# Global Gentle Harpsichord samples (loaded once)
GENTLEHARPSI_SAMPLES: List[Dict] = []

# Global Feedback samples (loaded once)
FEEDBACK_SAMPLES: List[Dict] = []

# Global Stylo samples (loaded once)
STYLO_SAMPLES: List[Dict] = []

# Global Trichords samples (loaded once)
TRICHORDS_SAMPLES: List[Dict] = []

# Global Tremolo Oct samples (loaded once)
TREMOLO_OCT_SAMPLES: List[Dict] = []

# Global Avery Violin samples (loaded once)
AVERYVIOLIN_SAMPLES: Dict[str, Dict] = {}

# Global Dictamel samples (loaded once)
DICTAMEL_SAMPLES: Dict[str, Dict] = {}

# Global Scelsi Pezzi samples (loaded once)
SCELSIPEZZI_SAMPLES: Dict[str, Dict] = {}

# Global Laken samples (loaded once)
LAKEN_SAMPLES: List[Dict] = []

# Global progression samples (multi-chord sequences)
GLAZ_SAX_SAMPLES: List[Dict] = []
HYACINTHE_SAMPLES: List[Dict] = []
KRAUS_SAMPLES: List[Dict] = []

# Global chord-triggered samples
GODETTE_SAMPLES: List[Dict] = []

# =============================================================================
# AUDIO CACHE - NumPy-based audio storage for fast processing
# =============================================================================
# Cache stores: {path_str: {"samples": np.ndarray, "sample_rate": int, "channels": int}}
AUDIO_CACHE: Dict[str, Dict] = {}


def get_cached_audio(audio_path: Path) -> Optional[Dict]:
    """
    Get audio from cache, loading it if not already cached.
    Returns dict with 'samples' (float32 numpy array shape [N, channels]),
    'sample_rate', and 'channels'.

    IMPORTANT: Samples are normalized to [-1.0, 1.0] range for mixing.
    Handles 8-bit, 16-bit, 24-bit, and 32-bit samples.
    Audio is resampled to TARGET_SAMPLE_RATE for consistent mixing.
    """
    global AUDIO_CACHE
    path_str = str(audio_path)

    if path_str in AUDIO_CACHE:
        return AUDIO_CACHE[path_str]

    if not audio_path.exists():
        return None

    try:
        audio = AudioSegment.from_file(audio_path)
        source_rate = audio.frame_rate
        raw_samples = audio.get_array_of_samples()
        samples = np.array(raw_samples, dtype=np.float32)

        # Normalize to [-1.0, 1.0] based on actual sample width
        sample_width = audio.sample_width
        if sample_width == 1:
            max_val = 128.0  # 8-bit
        elif sample_width == 2:
            max_val = 32768.0  # 16-bit
        elif sample_width == 3:
            max_val = 8388608.0  # 24-bit
        elif sample_width == 4:
            max_val = 2147483648.0  # 32-bit
        else:
            max_val = 32768.0  # fallback to 16-bit

        samples = samples / max_val

        channels = audio.channels
        if channels == 2:
            if len(samples) % 2 != 0:
                samples = samples[:-1]
            samples = samples.reshape((-1, 2))
        else:
            samples = samples.reshape((-1, 1))

        # Resample to TARGET_SAMPLE_RATE if needed (pitch-preserving)
        if source_rate != TARGET_SAMPLE_RATE:
            samples = resample_to_rate(samples, source_rate, TARGET_SAMPLE_RATE)

        AUDIO_CACHE[path_str] = {
            "samples": samples,
            "sample_rate": TARGET_SAMPLE_RATE,
            "channels": channels,
            "sample_width": sample_width
        }
        return AUDIO_CACHE[path_str]
    except Exception:
        return None


def apply_varispeed_np(samples: np.ndarray, semitones: float) -> np.ndarray:
    """
    Apply constant pitch shift via varispeed (vectorized).
    samples: float32 array shape [N, channels]
    Returns resampled array.
    """
    if abs(semitones) < 0.01:
        return samples

    rate = 2 ** (semitones / 12.0)
    num_input = len(samples)
    num_output = max(1, int(num_input / rate))
    channels = samples.shape[1]

    # Vectorized resampling
    output_positions = np.arange(num_output) * rate
    output_positions = np.clip(output_positions, 0, num_input - 1.001)
    input_indices = np.arange(num_input)

    output = np.zeros((num_output, channels), dtype=np.float32)
    for ch in range(channels):
        output[:, ch] = np.interp(output_positions, input_indices, samples[:, ch])

    return output


def resample_to_rate(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """
    Resample audio from source_rate to target_rate (pitch-preserving).
    samples: float32 array shape [N, channels]
    Returns resampled array at target_rate with same duration (different sample count).
    """
    if source_rate == target_rate:
        return samples

    ratio = target_rate / source_rate
    num_input = len(samples)
    num_output = max(1, int(num_input * ratio))
    channels = samples.shape[1]

    # Vectorized resampling using linear interpolation
    output_positions = np.arange(num_output) / ratio
    output_positions = np.clip(output_positions, 0, num_input - 1.001)
    input_indices = np.arange(num_input)

    output = np.zeros((num_output, channels), dtype=np.float32)
    for ch in range(channels):
        output[:, ch] = np.interp(output_positions, input_indices, samples[:, ch])

    return output


def apply_stereo_pan(samples: np.ndarray, pan: float) -> np.ndarray:
    """
    Apply stereo panning to audio.
    samples: float32 array shape [N, 2] (stereo)
    pan: -1.0 = full left, 0.0 = center, +1.0 = full right
    Returns panned stereo array.
    """
    if samples.shape[1] != 2:
        return samples

    # Clamp pan to valid range
    pan = np.clip(pan, -1.0, 1.0)

    # Equal power panning
    # At center (pan=0): both channels at ~0.707 (-3dB)
    # At hard left (pan=-1): left=1.0, right=0.0
    # At hard right (pan=+1): left=0.0, right=1.0
    angle = (pan + 1.0) * np.pi / 4.0  # 0 to pi/2
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)

    output = samples.copy()
    # Mix both input channels into each output channel with panning
    mono = (samples[:, 0] + samples[:, 1]) * 0.5
    output[:, 0] = mono * left_gain
    output[:, 1] = mono * right_gain

    return output


def apply_dynamic_pan_envelope(samples: np.ndarray, pan_values: np.ndarray) -> np.ndarray:
    """
    Apply a time-varying pan envelope to audio.
    samples: float32 array shape [N, 2] (stereo)
    pan_values: array of pan values (-1 to +1) of length N
    Returns panned stereo array.
    """
    if samples.shape[1] != 2 or len(pan_values) != len(samples):
        return samples

    # Clamp pan values
    pan_values = np.clip(pan_values, -1.0, 1.0)

    # Calculate per-sample gains using equal power panning
    angles = (pan_values + 1.0) * np.pi / 4.0
    left_gains = np.cos(angles)
    right_gains = np.sin(angles)

    # Mix to mono then redistribute
    mono = (samples[:, 0] + samples[:, 1]) * 0.5

    output = np.zeros_like(samples)
    output[:, 0] = mono * left_gains
    output[:, 1] = mono * right_gains

    return output


def detect_audio_activity(buffer: np.ndarray, window_ms: int = 50,
                          sample_rate: int = 44100, threshold: float = 0.001) -> np.ndarray:
    """
    Detect where audio is active (has content above threshold).
    Returns a boolean array of same length as buffer.
    """
    if len(buffer) == 0:
        return np.array([], dtype=bool)

    # Calculate RMS in windows
    window_samples = int(window_ms * sample_rate / 1000)
    if window_samples < 1:
        window_samples = 1

    # Get mono amplitude
    if len(buffer.shape) > 1:
        mono = np.abs(buffer).max(axis=1)
    else:
        mono = np.abs(buffer)

    # Pad to make length divisible by window
    pad_len = (window_samples - len(mono) % window_samples) % window_samples
    if pad_len > 0:
        mono = np.pad(mono, (0, pad_len), mode='constant')

    # Reshape and calculate max per window
    num_windows = len(mono) // window_samples
    windowed = mono[:num_windows * window_samples].reshape(num_windows, window_samples)
    window_maxes = windowed.max(axis=1)

    # Expand back to sample resolution
    activity = np.repeat(window_maxes > threshold, window_samples)[:len(buffer)]

    return activity


def apply_glissando_np(samples: np.ndarray, sample_rate: int,
                       semitones_start: float, semitones_end: float,
                       gliss_ms: int = GLISSANDO_MS) -> np.ndarray:
    """
    Apply pitch glissando using vectorized varispeed (NumPy-native).
    samples: float32 array shape [N, channels]
    Returns resampled array with glissando applied.
    """
    if abs(semitones_start - semitones_end) < 0.01:
        return apply_varispeed_np(samples, semitones_end) if abs(semitones_end) > 0.01 else samples

    num_input_samples = len(samples)
    channels = samples.shape[1]

    if num_input_samples < 2:
        return samples

    gliss_samples = int(gliss_ms * sample_rate / 1000)
    gliss_samples = min(gliss_samples, num_input_samples - 1)
    if gliss_samples < 1:
        return samples

    # Estimate output length for glissando portion
    avg_rate = (2 ** (semitones_start / 12.0) + 2 ** (semitones_end / 12.0)) / 2
    if avg_rate <= 0:
        avg_rate = 1.0
    output_gliss_samples = max(1, int(gliss_samples / avg_rate))

    # Build progress, easing, and rate curves (vectorized)
    progress = np.linspace(0, 1, output_gliss_samples)
    eased = np.sin(progress * np.pi / 2)
    semitone_curve = semitones_start + (semitones_end - semitones_start) * eased
    rate_curve = 2 ** (semitone_curve / 12.0)

    # Cumsum for input positions
    input_positions = np.concatenate([[0], np.cumsum(rate_curve[:-1])])
    input_positions = np.clip(input_positions, 0, num_input_samples - 1.001)

    # Resample glissando portion
    input_indices = np.arange(num_input_samples)
    output_gliss = np.zeros((output_gliss_samples, channels), dtype=np.float32)
    for ch in range(channels):
        output_gliss[:, ch] = np.interp(input_positions, input_indices, samples[:, ch])

    # Process rest of audio with final transposition
    final_input_pos = input_positions[-1] + rate_curve[-1] if output_gliss_samples > 0 else 0
    rest_start_input = int(final_input_pos)

    output_parts = [output_gliss]

    if rest_start_input < num_input_samples:
        rest_samples = samples[rest_start_input:]
        if len(rest_samples) > 0:
            rest_resampled = apply_varispeed_np(rest_samples, semitones_end)
            output_parts.append(rest_resampled)

    return np.vstack(output_parts)


def mix_into_buffer(buffer: np.ndarray, samples: np.ndarray,
                    start_sample: int, gain_db: float = 0.0) -> None:
    """
    Mix samples into buffer at the specified position (in-place).
    buffer: float32 array shape [total_samples, channels]
    samples: float32 array shape [N, channels]
    gain_db: dB adjustment to apply
    """
    if len(samples) == 0:
        return

    # Convert dB to linear gain
    gain = apply_gain_db(gain_db)

    end_sample = start_sample + len(samples)
    if end_sample > len(buffer):
        # Truncate samples to fit
        samples = samples[:len(buffer) - start_sample]
        end_sample = len(buffer)

    if start_sample >= len(buffer):
        return

    # Mix with gain
    buffer[start_sample:end_sample] += samples * gain


def buffer_to_audiosegment(buffer: np.ndarray, sample_rate: int) -> AudioSegment:
    """
    Convert float32 numpy buffer to AudioSegment.
    buffer: shape [N, channels], values in normalized [-1.0, 1.0] range
    """
    channels = buffer.shape[1] if len(buffer.shape) > 1 else 1

    # Clip to [-1.0, 1.0] and convert to int16 scale
    clipped = np.clip(buffer, -1.0, 1.0)
    int16_samples = (clipped * 32767.0).astype(np.int16)

    # Flatten for pydub (interleaved stereo)
    flat = int16_samples.flatten()

    # Ensure even length for stereo
    if channels == 2 and len(flat) % 2 != 0:
        flat = flat[:-1]

    return AudioSegment(
        data=flat.tobytes(),
        sample_width=2,  # int16
        frame_rate=sample_rate,
        channels=channels
    )


def build_harmonic_index(harmonic_timeline: List['HarmonicEvent']) -> Tuple[List[float], List['HarmonicEvent']]:
    """
    Build a sorted index for fast harmonic event lookup using bisect.
    Returns (start_times, events) where start_times[i] corresponds to events[i].
    Use bisect.bisect_right(start_times, time) - 1 to find the active event at 'time'.
    """
    # Sort by start time (should already be sorted, but ensure it)
    sorted_events = sorted(harmonic_timeline, key=lambda e: e.start_ms)
    start_times = [e.start_ms for e in sorted_events]
    return start_times, sorted_events


def get_harmonic_event_at(start_times: List[float], events: List['HarmonicEvent'],
                          time_ms: float) -> Optional['HarmonicEvent']:
    """
    Find the harmonic event active at the given time using bisect.
    O(log n) lookup instead of linear scan.
    """
    if not start_times:
        return None

    idx = bisect.bisect_right(start_times, time_ms) - 1
    if idx < 0:
        return events[0] if events else None

    event = events[idx]
    # Verify time is within this event's range
    if event.start_ms <= time_ms < event.end_ms:
        return event
    elif time_ms >= event.end_ms and idx + 1 < len(events):
        # Time is after this event but before next - return this event
        return event

    return event


def render_constant_rate_output_driven(
    samples: np.ndarray,
    input_pos: float,
    output_len: int,
    rate: float,
    channels: int
) -> Tuple[np.ndarray, float]:
    """
    Render a constant-rate segment with OUTPUT length as the constraint.

    Args:
        samples: Source audio [N, channels]
        input_pos: Current position in input samples
        output_len: Desired number of OUTPUT samples
        rate: Playback rate (2^(semitones/12))
        channels: Number of audio channels

    Returns:
        (rendered_output, new_input_pos)
    """
    num_input = len(samples)

    # Generate output sample indices
    output_indices = np.arange(output_len)

    # Map to input positions: input_pos advances by 'rate' per output sample
    input_positions = input_pos + output_indices * rate

    # Find where we exceed input bounds
    valid_mask = input_positions < num_input - 1
    valid_len = np.sum(valid_mask)

    if valid_len == 0:
        return np.zeros((0, channels), dtype=np.float32), input_pos

    input_positions = input_positions[:valid_len]

    # Linear interpolation
    floor_idx = np.floor(input_positions).astype(int)
    ceil_idx = np.minimum(floor_idx + 1, num_input - 1)
    fracs = (input_positions - floor_idx).reshape(-1, 1)

    result = samples[floor_idx] * (1 - fracs) + samples[ceil_idx] * fracs

    # Return new input position
    new_input_pos = input_pos + valid_len * rate

    return result, new_input_pos


def render_glissando_output_driven(
    samples: np.ndarray,
    input_pos: float,
    output_len: int,
    start_trans: float,
    end_trans: float,
    channels: int
) -> Tuple[np.ndarray, float]:
    """
    Render a glissando segment with OUTPUT length as the constraint.
    Uses quarter-sine easing for smooth pitch transition.

    Args:
        samples: Source audio [N, channels]
        input_pos: Current position in input samples
        output_len: Desired number of OUTPUT samples (fixed by glissando_ms)
        start_trans: Starting transposition in semitones
        end_trans: Ending transposition in semitones
        channels: Number of audio channels

    Returns:
        (rendered_output, new_input_pos)
    """
    num_input = len(samples)

    if output_len <= 0:
        return np.zeros((0, channels), dtype=np.float32), input_pos

    # Build rate curve with quarter-sine easing over OUTPUT samples
    progress = np.linspace(0, 1, output_len)
    eased = np.sin(progress * np.pi / 2)
    trans_curve = start_trans + (end_trans - start_trans) * eased
    rate_curve = 2 ** (trans_curve / 12.0)

    # Compute input positions by integrating rate over output samples
    # input_pos[i+1] = input_pos[i] + rate[i]
    cumulative_input = np.cumsum(rate_curve)
    input_positions = input_pos + np.insert(cumulative_input, 0, 0)[:-1]

    # Find where we exceed input bounds
    valid_mask = input_positions < num_input - 1
    valid_len = np.sum(valid_mask)

    if valid_len == 0:
        return np.zeros((0, channels), dtype=np.float32), input_pos

    input_positions = input_positions[:valid_len]

    # Linear interpolation
    floor_idx = np.floor(input_positions).astype(int)
    ceil_idx = np.minimum(floor_idx + 1, num_input - 1)
    fracs = (input_positions - floor_idx).reshape(-1, 1)

    result = samples[floor_idx] * (1 - fracs) + samples[ceil_idx] * fracs

    # Return new input position (sum of all rates consumed)
    new_input_pos = input_pos + cumulative_input[valid_len - 1] if valid_len > 0 else input_pos

    return result, new_input_pos


def apply_varispeed_segment_np(samples: np.ndarray, semitones: float) -> np.ndarray:
    """Apply constant varispeed to a segment using vectorized NumPy."""
    if semitones == 0 or len(samples) == 0:
        return samples.copy()

    rate = 2 ** (semitones / 12.0)
    original_len = len(samples)
    new_len = int(original_len / rate)

    if new_len < 1:
        return samples[:1].copy()

    old_indices = np.linspace(0, original_len - 1, new_len)
    old_indices_floor = np.floor(old_indices).astype(int)
    old_indices_ceil = np.minimum(old_indices_floor + 1, original_len - 1)
    fracs = (old_indices - old_indices_floor).reshape(-1, 1)

    return samples[old_indices_floor] * (1 - fracs) + samples[old_indices_ceil] * fracs


def apply_glissando_segment_np(
    samples: np.ndarray,
    start_trans: float,
    end_trans: float,
    sample_rate: int
) -> np.ndarray:
    """
    Apply glissando (pitch glide) to a segment using vectorized NumPy.
    Uses quarter-sine easing like the original.
    """
    if len(samples) == 0:
        return samples.copy()

    num_input = len(samples)
    channels = samples.shape[1] if samples.ndim > 1 else 1

    # Calculate output length based on average rate
    avg_rate = 2 ** ((start_trans + end_trans) / 2 / 12.0)
    estimated_output_len = int(num_input / avg_rate) + 100

    # Build rate curve with quarter-sine easing
    progress = np.linspace(0, 1, estimated_output_len)
    eased = np.sin(progress * np.pi / 2)
    trans_curve = start_trans + (end_trans - start_trans) * eased
    rate_curve = 2 ** (trans_curve / 12.0)

    # Cumulative input position
    input_positions = np.cumsum(rate_curve)
    input_positions = np.insert(input_positions, 0, 0)[:-1]

    # Find where we exceed input length
    valid_mask = input_positions < num_input - 1
    valid_len = np.sum(valid_mask)
    if valid_len == 0:
        return samples[:1].copy()

    input_positions = input_positions[:valid_len]

    # Interpolate
    floor_idx = np.floor(input_positions).astype(int)
    ceil_idx = np.minimum(floor_idx + 1, num_input - 1)
    fracs = (input_positions - floor_idx).reshape(-1, 1)

    return samples[floor_idx] * (1 - fracs) + samples[ceil_idx] * fracs


def render_overlay_reactive_np(
    samples: np.ndarray,
    sample_rate: int,
    original_pcs: Set[int],
    start_ms: float,
    harmonic_start_times: List[float],
    harmonic_events: List['HarmonicEvent'],
    find_transposition_fn
) -> Tuple[np.ndarray, int]:
    """
    Reactive varispeed with glissando - OUTPUT TIME scheduling.

    CRITICAL: All timing is in OUTPUT/render time. Events, glides, and chord
    changes are scheduled in rendered time. Only input_pos advances at varispeed.

    Architecture:
    1. Build schedule in OUTPUT sample space
    2. For each output segment: compute input consumption via rate integration
    3. Render with vectorized NumPy

    Returns (processed_samples, gliss_count)
    """
    if len(samples) == 0:
        return samples, 0

    num_input_samples = len(samples)
    channels = samples.shape[1]
    glissando_samples = int(GLISSANDO_MS * sample_rate / 1000)  # OUTPUT samples

    # === STEP 1: Find initial transposition ===
    onset_idx = bisect.bisect_right(harmonic_start_times, start_ms) - 1
    if onset_idx < 0:
        onset_idx = 0
    if onset_idx >= len(harmonic_events):
        onset_idx = len(harmonic_events) - 1

    initial_he = harmonic_events[onset_idx]
    initial_trans = find_transposition_fn(original_pcs, initial_he.chord_pcs, initial_he.chord_root)
    if initial_trans is None:
        initial_trans = 0

    # === STEP 2: Build OUTPUT-time schedule ===
    # Each entry: (output_start_sample, output_end_sample, start_trans, end_trans, is_glide)
    schedule = []
    gliss_count = 0

    # Estimate max output duration (will be refined during rendering)
    avg_rate_estimate = 2 ** (initial_trans / 12.0)
    max_output_samples = int(num_input_samples / avg_rate_estimate * 1.5)

    # Collect harmonic events during playback (in output ms from sample start)
    events_during = []  # [(output_ms_offset, transposition)]
    estimated_duration_ms = num_input_samples * 1000.0 / sample_rate / avg_rate_estimate
    end_time_ms = start_ms + estimated_duration_ms * 1.5

    for i in range(onset_idx + 1, len(harmonic_events)):
        he = harmonic_events[i]
        if he.start_ms > end_time_ms:
            break
        if he.start_ms > start_ms:
            trans = find_transposition_fn(original_pcs, he.chord_pcs, he.chord_root)
            if trans is None:
                trans = initial_trans
            events_during.append((he.start_ms - start_ms, trans))

    # Build schedule in OUTPUT sample space
    current_trans = float(initial_trans)
    current_output_sample = 0

    for event_ms_offset, next_trans in events_during:
        event_output_sample = int(event_ms_offset * sample_rate / 1000)

        if event_output_sample <= current_output_sample:
            current_trans = float(next_trans)
            continue

        # Constant-rate segment up to event
        schedule.append((
            current_output_sample,
            event_output_sample,
            current_trans,
            current_trans,
            False
        ))

        # Glissando segment (fixed duration in OUTPUT time)
        if next_trans != current_trans:
            gliss_count += 1
            gliss_end = event_output_sample + glissando_samples
            schedule.append((
                event_output_sample,
                gliss_end,
                current_trans,
                float(next_trans),
                True
            ))
            current_output_sample = gliss_end
        else:
            current_output_sample = event_output_sample

        current_trans = float(next_trans)

    # Final segment - will be trimmed when input runs out
    schedule.append((
        current_output_sample,
        max_output_samples,
        current_trans,
        current_trans,
        False
    ))

    # === STEP 3: Render each segment ===
    # Track input position across segments
    output_segments = []
    input_pos = 0.0

    for out_start, out_end, start_trans, end_trans, is_glide in schedule:
        if input_pos >= num_input_samples:
            break

        out_len = out_end - out_start
        if out_len <= 0:
            continue

        if is_glide:
            # Glissando: varying rate over output samples
            rendered, new_input_pos = render_glissando_output_driven(
                samples, input_pos, out_len, start_trans, end_trans, channels
            )
        else:
            # Constant rate
            rate = 2 ** (start_trans / 12.0)
            rendered, new_input_pos = render_constant_rate_output_driven(
                samples, input_pos, out_len, rate, channels
            )

        if len(rendered) > 0:
            output_segments.append(rendered)

        input_pos = new_input_pos
        if input_pos >= num_input_samples:
            break

    # === STEP 4: Concatenate ===
    if not output_segments:
        return np.zeros((0, channels), dtype=np.float32), gliss_count

    result = np.concatenate(output_segments, axis=0)
    return result, gliss_count


def render_sample_reactive_np(
    audio_path: Path,
    sample_pcs: Set[int],
    start_ms: float,
    harmonic_start_times: List[float],
    harmonic_events: List['HarmonicEvent'],
    find_transposition_fn
) -> Tuple[np.ndarray, int, int]:
    """
    Render a sample with reactive transposition using NumPy.
    Uses audio cache for fast loading.

    Returns (samples, sample_rate, gliss_count)
    """
    cached = get_cached_audio(audio_path)
    if cached is None:
        return np.zeros((0, 2), dtype=np.float32), 44100, 0

    samples = cached["samples"]
    sample_rate = cached["sample_rate"]

    result, gliss_count = render_overlay_reactive_np(
        samples, sample_rate, sample_pcs, start_ms,
        harmonic_start_times, harmonic_events, find_transposition_fn
    )

    return result, sample_rate, gliss_count


def render_layer_event_driven_np(
    sample_list: List[Dict],
    harmonic_start_times: List[float],
    harmonic_events: List['HarmonicEvent'],
    find_transposition_fn,
    total_duration_ms: float,
    sample_rate: int = 44100,
    channels: int = 2,
    gain_db: float = 0.0,
    verbose: bool = False,
    layer_name: str = "Layer",
    glissando_ms: float = 150.0
) -> np.ndarray:
    """
    Render a continuous layer with EVENT-DRIVEN reactive transposition.

    Reactive but fast:
    - At note start: find initial transposition
    - Precompute harmonic change events that occur during the sample
    - Convert events to output sample indices
    - During rendering: retarget glide only when event index is reached
    - If event arrives mid-glide: retarget from current interpolated position

    NO per-sample harmonic polling. Reactivity stays, polling dies.
    """
    total_samples = int(total_duration_ms * sample_rate / 1000)
    buffer = np.zeros((total_samples, channels), dtype=np.float32)

    if not sample_list:
        return buffer

    glissando_samples = int(glissando_ms * sample_rate / 1000)
    current_output_sample = 0
    sample_idx = 0
    num_samples = len(sample_list)
    played_count = 0
    total_glissandos = 0

    while current_output_sample < total_samples:
        sample_data = sample_list[sample_idx % num_samples]
        sample_idx += 1

        audio_path = sample_data.get("audio_path")
        sample_pcs = sample_data.get("pitch_classes", set())

        if audio_path is None:
            continue

        # Load source audio
        cached = get_cached_audio(Path(audio_path))
        if cached is None:
            continue

        source_samples = cached["samples"]
        source_len = len(source_samples)
        source_sr = cached.get("sample_rate", sample_rate)

        # Calculate sample timing
        onset_ms = current_output_sample * 1000.0 / sample_rate
        # Estimate duration based on source length (will vary with transposition)
        estimated_duration_ms = source_len * 1000.0 / source_sr

        # Find initial harmonic event at onset
        onset_idx = bisect.bisect_right(harmonic_start_times, onset_ms) - 1
        if onset_idx < 0:
            onset_idx = 0
        if onset_idx >= len(harmonic_events):
            onset_idx = len(harmonic_events) - 1

        initial_he = harmonic_events[onset_idx]
        initial_trans = find_transposition_fn(sample_pcs, initial_he.chord_pcs, initial_he.chord_root)
        if initial_trans is None:
            initial_trans = 0  # Default to no transposition if can't find fit

        # Precompute harmonic events during this sample's playback window
        # Find all events between onset_ms and onset_ms + estimated_duration_ms
        scheduled_events = []  # [(output_sample_idx, transposition)]

        end_ms = onset_ms + estimated_duration_ms * 1.5  # Add margin for rate changes
        for i in range(onset_idx + 1, len(harmonic_events)):
            he = harmonic_events[i]
            if he.start_ms > end_ms:
                break
            # Convert event time to output sample index relative to this sample's start
            event_output_idx = int((he.start_ms - onset_ms) * sample_rate / 1000)
            if event_output_idx > 0:
                trans = find_transposition_fn(sample_pcs, he.chord_pcs, he.chord_root)
                if trans is None:
                    trans = 0
                scheduled_events.append((event_output_idx, trans))
                total_glissandos += 1

        # Render this sample with event-driven transposition
        # Build the output by processing source samples with variable rate
        current_trans = float(initial_trans)
        target_trans = float(initial_trans)
        gliss_progress = glissando_samples  # Start fully converged

        output_samples = []
        source_pos = 0.0
        event_ptr = 0
        local_output_idx = 0

        while source_pos < source_len - 1:
            # Check if we've hit a scheduled event
            if event_ptr < len(scheduled_events):
                event_idx, event_trans = scheduled_events[event_ptr]
                if local_output_idx >= event_idx:
                    # New target - retarget from current position
                    target_trans = float(event_trans)
                    gliss_progress = 0
                    event_ptr += 1

            # Update current transposition (glide toward target)
            if gliss_progress < glissando_samples:
                t = gliss_progress / glissando_samples
                current_trans = current_trans + (target_trans - current_trans) * min(t, 1.0)
                gliss_progress += 1

            # Calculate playback rate
            rate = 2 ** (current_trans / 12.0)

            # Interpolate source sample
            idx_floor = int(source_pos)
            idx_ceil = min(idx_floor + 1, source_len - 1)
            frac = source_pos - idx_floor
            sample_val = source_samples[idx_floor] * (1 - frac) + source_samples[idx_ceil] * frac
            output_samples.append(sample_val)

            source_pos += rate
            local_output_idx += 1

            # Safety limit
            if local_output_idx > total_samples * 2:
                break

        if len(output_samples) == 0:
            continue

        played_count += 1
        result = np.array(output_samples, dtype=np.float32)

        # Mix into buffer
        end_sample = min(current_output_sample + len(result), total_samples)
        usable_len = end_sample - current_output_sample

        if usable_len > 0:
            gain = apply_gain_db(gain_db)
            # Handle channel mismatch
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            if result.shape[1] != channels:
                if channels == 2 and result.shape[1] == 1:
                    result = np.hstack([result, result])
                elif channels == 1 and result.shape[1] == 2:
                    result = result.mean(axis=1, keepdims=True)
            buffer[current_output_sample:end_sample] += result[:usable_len] * gain

        current_output_sample += len(result)

    if verbose:
        print(f"    {layer_name}: {played_count} samples played continuously, {total_glissandos} glissandos")

    return buffer


def render_layer_onset_transposed_np(
    sample_list: List[Dict],
    harmonic_start_times: List[float],
    harmonic_events: List['HarmonicEvent'],
    find_transposition_fn,
    total_duration_ms: float,
    sample_rate: int = 44100,
    channels: int = 2,
    gain_db: float = 0.0,
    verbose: bool = False,
    layer_name: str = "Layer"
) -> np.ndarray:
    """
    Render a continuous layer using ONSET-ONLY transposition (fast).

    Each sample:
    - Looks up chord at onset time (ONE binary search)
    - Calculates transposition once
    - Applies constant varispeed
    - Mixes into buffer

    NO per-sample reactive rendering. NO gliding mid-sample.
    This is 10-100x faster than render_layer_continuous_np.
    """
    total_samples = int(total_duration_ms * sample_rate / 1000)
    buffer = np.zeros((total_samples, channels), dtype=np.float32)

    if not sample_list:
        return buffer

    current_sample = 0
    sample_idx = 0
    num_samples = len(sample_list)
    played_count = 0

    while current_sample < total_samples:
        sample_data = sample_list[sample_idx % num_samples]
        sample_idx += 1

        audio_path = sample_data.get("audio_path")
        sample_pcs = sample_data.get("pitch_classes", set())

        if audio_path is None:
            continue

        # Load audio
        cached = get_cached_audio(Path(audio_path))
        if cached is None:
            continue

        samples = cached["samples"].copy()

        # Get onset time in ms
        onset_ms = current_sample * 1000.0 / sample_rate

        # Find harmonic event at onset (ONE binary search)
        idx = bisect.bisect_right(harmonic_start_times, onset_ms) - 1
        if idx < 0:
            idx = 0
        if idx >= len(harmonic_events):
            idx = len(harmonic_events) - 1

        he = harmonic_events[idx]

        # Calculate transposition once
        trans = find_transposition_fn(sample_pcs, he.chord_pcs, he.chord_root)

        # Apply constant varispeed if needed
        if trans != 0:
            rate = 2 ** (trans / 12.0)
            original_len = len(samples)
            new_len = int(original_len / rate)
            if new_len > 0:
                old_indices = np.linspace(0, original_len - 1, new_len)
                old_indices_floor = np.floor(old_indices).astype(int)
                old_indices_ceil = np.minimum(old_indices_floor + 1, original_len - 1)
                fracs = (old_indices - old_indices_floor).reshape(-1, 1)
                samples = samples[old_indices_floor] * (1 - fracs) + samples[old_indices_ceil] * fracs

        played_count += 1

        # Mix into buffer
        end_sample = min(current_sample + len(samples), total_samples)
        usable_len = end_sample - current_sample

        if usable_len > 0:
            gain = apply_gain_db(gain_db)
            # Handle channel mismatch
            if samples.shape[1] != channels:
                if channels == 2 and samples.shape[1] == 1:
                    samples = np.hstack([samples, samples])
                elif channels == 1 and samples.shape[1] == 2:
                    samples = samples.mean(axis=1, keepdims=True)
            buffer[current_sample:end_sample] += samples[:usable_len] * gain

        current_sample += len(samples)

    if verbose:
        print(f"    {layer_name}: {played_count} samples played continuously")

    return buffer


def render_layer_continuous_np(
    sample_list: List[Dict],
    harmonic_start_times: List[float],
    harmonic_events: List['HarmonicEvent'],
    find_transposition_fn,
    total_duration_ms: float,
    sample_rate: int = 44100,
    channels: int = 2,
    gain_db: float = 0.0,
    verbose: bool = False,
    layer_name: str = "Layer"
) -> np.ndarray:
    """
    Render a continuous layer (like bass flute or jicello) using NumPy.
    Samples play back-to-back with no gaps.

    Returns float32 buffer [total_samples, channels]
    """
    total_samples = int(total_duration_ms * sample_rate / 1000)
    buffer = np.zeros((total_samples, channels), dtype=np.float32)

    if not sample_list:
        return buffer

    current_sample = 0
    sample_idx = 0
    num_samples = len(sample_list)
    played_count = 0

    while current_sample < total_samples:
        sample_data = sample_list[sample_idx % num_samples]
        sample_idx += 1

        audio_path = sample_data.get("audio_path")
        sample_pcs = sample_data.get("pitch_classes", set())

        if audio_path is None:
            continue

        start_ms = current_sample * 1000.0 / sample_rate

        result, _, gliss_count = render_sample_reactive_np(
            audio_path, sample_pcs, start_ms,
            harmonic_start_times, harmonic_events, find_transposition_fn
        )

        if len(result) == 0:
            # Skip to next sample
            continue

        played_count += 1

        # Mix into buffer at current position
        end_sample = min(current_sample + len(result), total_samples)
        usable_len = end_sample - current_sample

        if usable_len > 0:
            gain = apply_gain_db(gain_db)
            # Handle channel mismatch
            if result.shape[1] != channels:
                if channels == 2 and result.shape[1] == 1:
                    result = np.hstack([result, result])
                elif channels == 1 and result.shape[1] == 2:
                    result = result.mean(axis=1, keepdims=True)
            buffer[current_sample:end_sample] += result[:usable_len] * gain

        # Move to end of this sample
        current_sample += len(result)

    if verbose:
        print(f"    {layer_name}: {played_count} samples played continuously")

    return buffer


def render_layer_interval_np(
    sample_list: List[Dict],
    harmonic_start_times: List[float],
    harmonic_events: List['HarmonicEvent'],
    find_transposition_fn,
    total_duration_ms: float,
    interval_seconds: float = 4.0,
    sample_rate: int = 44100,
    channels: int = 2,
    gain_db: float = 0.0,
    verbose: bool = False,
    layer_name: str = "Layer",
    skip_to_fitting: bool = True,
    random_selection: bool = False,
    seed: Optional[int] = None,
    max_overlap: Optional[int] = None
) -> np.ndarray:
    """
    Render an interval-based layer (like Organetta) using NumPy.
    Samples fire at fixed intervals. Uses event-driven reactive transposition.

    skip_to_fitting: If True, skips to next fitting sample when current can't fit.
    random_selection: If True, selects samples randomly instead of sequentially.
    seed: Random seed for reproducible random selection.
    max_overlap: Maximum number of samples that can play simultaneously.
                 If None, no limit. Skips firing if limit would be exceeded.

    Returns float32 buffer [total_samples, channels]
    """
    if random_selection and seed is not None:
        random.seed(seed + 777)  # Offset for variety

    total_samples = int(total_duration_ms * sample_rate / 1000)
    buffer = np.zeros((total_samples, channels), dtype=np.float32)

    if not sample_list:
        return buffer

    interval_ms = interval_seconds * 1000.0
    num_fires = int(total_duration_ms / interval_ms)

    if verbose:
        mode_str = "random" if random_selection else "sequential"
        overlap_str = f", max {max_overlap} overlap" if max_overlap else ""
        print(f"    {layer_name}: {len(sample_list)} samples, every {interval_seconds}s, {num_fires} total slots ({mode_str}{overlap_str})")
        if skip_to_fitting:
            print(f"    (Reactive transposition + skip-to-fitting-sample enabled)")

    played_count = 0
    skipped_count = 0
    overlap_skipped = 0
    gliss_total = 0
    next_sample_idx = 0
    num_samples = len(sample_list)

    # Track end times of currently playing samples for max_overlap
    active_end_samples: List[int] = []

    for i in range(num_fires):
        start_ms = i * interval_ms
        start_sample = int(start_ms * sample_rate / 1000)

        if start_sample >= total_samples:
            break

        # Check overlap limit if set
        if max_overlap is not None:
            # Remove samples that have finished
            active_end_samples = [end for end in active_end_samples if end > start_sample]
            if len(active_end_samples) >= max_overlap:
                overlap_skipped += 1
                continue

        # Get current harmonic state
        he = get_harmonic_event_at(harmonic_start_times, harmonic_events, start_ms)
        if he is None:
            skipped_count += 1
            continue

        # Find a sample that fits (if skip_to_fitting)
        found_sample = None
        if random_selection:
            # Random selection - pick randomly, try to find one that fits
            if skip_to_fitting:
                shuffled_indices = list(range(num_samples))
                random.shuffle(shuffled_indices)
                for idx in shuffled_indices:
                    sample_data = sample_list[idx]
                    sample_pcs = sample_data.get("pitch_classes", set())
                    trans = find_transposition_fn(sample_pcs, he.chord_pcs, he.chord_root)
                    if trans is not None:
                        found_sample = sample_data
                        break
            else:
                found_sample = random.choice(sample_list)
        elif skip_to_fitting:
            for attempt in range(num_samples):
                idx = (next_sample_idx + attempt) % num_samples
                sample_data = sample_list[idx]
                sample_pcs = sample_data.get("pitch_classes", set())
                trans = find_transposition_fn(sample_pcs, he.chord_pcs, he.chord_root)
                if trans is not None:
                    found_sample = sample_data
                    next_sample_idx = (idx + 1) % num_samples
                    break
        else:
            found_sample = sample_list[next_sample_idx % num_samples]
            next_sample_idx = (next_sample_idx + 1) % num_samples

        if found_sample is None:
            skipped_count += 1
            continue

        audio_path = found_sample.get("audio_path")
        sample_pcs = found_sample.get("pitch_classes", set())

        if audio_path is None:
            skipped_count += 1
            continue

        result, _, gliss_count = render_sample_reactive_np(
            audio_path, sample_pcs, start_ms,
            harmonic_start_times, harmonic_events, find_transposition_fn
        )

        if len(result) == 0:
            skipped_count += 1
            continue

        played_count += 1
        gliss_total += gliss_count

        # Overlay into buffer at start position
        end_sample = min(start_sample + len(result), total_samples)
        usable_len = end_sample - start_sample

        if usable_len > 0:
            gain = apply_gain_db(gain_db)
            # Handle channel mismatch
            if result.shape[1] != channels:
                if channels == 2 and result.shape[1] == 1:
                    result = np.tile(result, (1, 2))
                elif channels == 1 and result.shape[1] == 2:
                    result = result.mean(axis=1, keepdims=True)
            buffer[start_sample:end_sample] += result[:usable_len] * gain

            # Track this sample's end time for overlap limiting
            if max_overlap is not None:
                active_end_samples.append(start_sample + len(result))

    if verbose:
        overlap_str = f", {overlap_skipped} overlap-skipped" if max_overlap else ""
        print(f"    {layer_name}: {played_count} played, {skipped_count} dropped out{overlap_str}, {gliss_total} glissandos")

    return buffer


def render_layer_cloud_np(
    sample_list: List[Dict],
    harmonic_start_times: List[float],
    harmonic_events: List['HarmonicEvent'],
    find_transposition_fn,
    total_duration_ms: float,
    min_silence_seconds: float = 10.0,
    max_silence_seconds: float = 15.0,
    cloud_duration_seconds: float = 4.0,
    bpm: float = 120.0,
    note_division: int = 16,
    sample_rate: int = 44100,
    channels: int = 2,
    gain_db: float = 0.0,
    verbose: bool = False,
    layer_name: str = "Cloud",
    random_selection: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Render a cloud-based layer (like Brodero, GothicHarp) using NumPy.
    Sporadic bursts of rapid notes with random silence between.

    The pattern is:
    1. Start with random silence (min_silence to max_silence seconds)
    2. Play a cloud of rapid notes for cloud_duration_seconds
    3. Random silence
    4. Repeat until end

    Returns float32 buffer [total_samples, channels]
    """
    if seed is not None:
        random.seed(seed + 888)  # Different seed offset for variety

    total_samples = int(total_duration_ms * sample_rate / 1000)
    buffer = np.zeros((total_samples, channels), dtype=np.float32)

    if not sample_list:
        return buffer

    # Calculate note interval based on BPM and division
    beat_ms = 60000.0 / bpm  # Quarter note in ms
    note_ms = beat_ms / (note_division / 4)  # e.g., 16th note = beat/4

    # Notes per cloud
    cloud_duration_ms = cloud_duration_seconds * 1000.0
    notes_per_cloud = int(cloud_duration_ms / note_ms)

    num_samples = len(sample_list)

    if verbose:
        note_name = {4: "quarter", 8: "8th", 16: "16th", 32: "32nd"}.get(note_division, f"1/{note_division}")
        print(f"    {layer_name}: {num_samples} samples")
        print(f"    Clouds: ~{notes_per_cloud} {note_name} notes ({cloud_duration_seconds}s) after {min_silence_seconds}-{max_silence_seconds}s silence")
        print(f"    ({note_name} note = {note_ms:.0f}ms at {bpm} BPM)")

    gliss_total = 0
    total_notes = 0
    num_clouds = 0
    next_sample_idx = 0

    # Start with random silence
    current_ms = random.uniform(min_silence_seconds, max_silence_seconds) * 1000.0

    while current_ms < total_duration_ms:
        cloud_start_ms = current_ms
        num_clouds += 1

        # Render notes in this cloud
        for note_idx in range(notes_per_cloud):
            note_start_ms = cloud_start_ms + (note_idx * note_ms)

            if note_start_ms >= total_duration_ms:
                break

            start_sample = int(note_start_ms * sample_rate / 1000)
            if start_sample >= total_samples:
                break

            # Get current harmonic state
            he = get_harmonic_event_at(harmonic_start_times, harmonic_events, note_start_ms)
            if he is None:
                continue

            # Select sample
            if random_selection:
                sample_data = random.choice(sample_list)
            else:
                sample_data = sample_list[next_sample_idx % num_samples]
                next_sample_idx += 1

            audio_path = sample_data.get("audio_path")
            sample_pcs = sample_data.get("pitch_classes", set())

            if audio_path is None:
                continue

            result, _, gliss_count = render_sample_reactive_np(
                audio_path, sample_pcs, note_start_ms,
                harmonic_start_times, harmonic_events, find_transposition_fn
            )

            if len(result) == 0:
                continue

            total_notes += 1
            gliss_total += gliss_count

            # Overlay into buffer
            end_sample = min(start_sample + len(result), total_samples)
            usable_len = end_sample - start_sample

            if usable_len > 0:
                gain = apply_gain_db(gain_db)
                # Handle channel mismatch
                if result.shape[1] != channels:
                    if channels == 2 and result.shape[1] == 1:
                        result = np.tile(result, (1, 2))
                    elif channels == 1 and result.shape[1] == 2:
                        result = result.mean(axis=1, keepdims=True)
                buffer[start_sample:end_sample] += result[:usable_len] * gain

        # Move past cloud, add random silence before next
        current_ms = cloud_start_ms + cloud_duration_ms
        current_ms += random.uniform(min_silence_seconds, max_silence_seconds) * 1000.0

    if verbose:
        print(f"    {layer_name}: {num_clouds} clouds, {total_notes} total notes, {gliss_total} glissandos")

    return buffer


def render_layer_chord_triggered_np(
    sample_list: List[Dict],
    harmonic_start_times: List[float],
    harmonic_events: List['HarmonicEvent'],
    find_transposition_fn,
    total_duration_ms: float,
    sample_rate: int = 44100,
    channels: int = 2,
    gain_db: float = 0.0,
    verbose: bool = False,
    layer_name: str = "Layer",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Render a chord-triggered layer (like Godette) using NumPy.

    Behavior:
    - Fires at the beginning of each new chord event
    - Only one sample can play at a time (no overlap)
    - Only plays if a fitting sample exists for the current chord
    - Cycles through samples sequentially when fitting samples are found

    Returns float32 buffer [total_samples, channels]
    """
    total_samples = int(total_duration_ms * sample_rate / 1000)
    buffer = np.zeros((total_samples, channels), dtype=np.float32)

    if not sample_list:
        return buffer

    if not harmonic_events:
        return buffer

    if verbose:
        print(f"    {layer_name}: {len(sample_list)} samples, {len(harmonic_events)} chord events")
        print(f"    (Chord-triggered: fires on chord changes if fitting sample exists)")

    played_count = 0
    skipped_count = 0
    gliss_total = 0
    next_sample_idx = 0
    num_samples = len(sample_list)

    # Track when current sample ends (for no-overlap rule)
    current_end_sample = 0

    for i, he in enumerate(harmonic_events):
        # Get start time from harmonic_start_times
        if i >= len(harmonic_start_times):
            break

        start_ms = harmonic_start_times[i]
        start_sample = int(start_ms * sample_rate / 1000)

        if start_sample >= total_samples:
            break

        # Check no-overlap rule: skip if previous sample is still playing
        if start_sample < current_end_sample:
            skipped_count += 1
            continue

        # Find a sample that fits the current chord
        found_sample = None
        for attempt in range(num_samples):
            idx = (next_sample_idx + attempt) % num_samples
            sample_data = sample_list[idx]
            sample_pcs = sample_data.get("pitch_classes", set())
            trans = find_transposition_fn(sample_pcs, he.chord_pcs, he.chord_root)
            if trans is not None:
                found_sample = sample_data
                next_sample_idx = (idx + 1) % num_samples
                break

        if found_sample is None:
            # No fitting sample found for this chord - skip silently
            skipped_count += 1
            continue

        audio_path = found_sample.get("audio_path")
        sample_pcs = found_sample.get("pitch_classes", set())

        if audio_path is None:
            skipped_count += 1
            continue

        result, _, gliss_count = render_sample_reactive_np(
            audio_path, sample_pcs, start_ms,
            harmonic_start_times, harmonic_events, find_transposition_fn
        )

        if len(result) == 0:
            skipped_count += 1
            continue

        played_count += 1
        gliss_total += gliss_count

        # Overlay into buffer at start position
        end_sample = min(start_sample + len(result), total_samples)
        usable_len = end_sample - start_sample

        if usable_len > 0:
            gain = apply_gain_db(gain_db)
            # Handle channel mismatch
            if result.shape[1] != channels:
                if channels == 2 and result.shape[1] == 1:
                    result = np.tile(result, (1, 2))
                elif channels == 1 and result.shape[1] == 2:
                    result = result.mean(axis=1, keepdims=True)
            buffer[start_sample:end_sample] += result[:usable_len] * gain

            # Update end time for no-overlap tracking
            current_end_sample = end_sample

    if verbose:
        print(f"    {layer_name}: {played_count} played, {skipped_count} skipped (no fit or overlap), {gliss_total} glissandos")

    return buffer


def load_chord_dictionary() -> Dict[str, Dict]:
    """Load the chord dictionary from chords_no_supersets.json."""
    global CHORD_DICTIONARY
    if CHORD_DICTIONARY:
        return CHORD_DICTIONARY

    with open(CHORDS_JSON_PATH) as f:
        raw_chords = json.load(f)

    # Process each chord to compute actual pitch classes
    for chord_key, chord_data in raw_chords.items():
        root = chord_data.get("root", 0)
        # NOTE: "prime_form_kinda" is a misnomer - it's actually intervals from root,
        # NOT true set-theory prime form. We transpose these intervals by the root
        # to get actual pitch classes (0-11). No normalization is applied.
        intervals_from_root = chord_data.get("prime_form_kinda", [])
        original_voicing = chord_data.get("original_voicing", [])

        # Compute actual pitch classes: (root + interval) % 12
        # This gives us octave-equivalent pitch classes, NOT normalized prime form
        # IMPORTANT: Always include the root pitch class, even if not in intervals
        # (voicings may only contain upper structure, omitting the root)
        actual_pcs = set((root + interval) % 12 for interval in intervals_from_root)
        actual_pcs.add(root)  # Ensure root is always included
        actual_pcs = sorted(actual_pcs)

        CHORD_DICTIONARY[chord_key] = {
            "name": chord_key,
            "chord_type": chord_data.get("chord_type", ""),
            "root": root,
            "intervals_from_root": intervals_from_root,  # NOT true prime form
            "pitch_classes": actual_pcs,  # Actual pitch classes 0-11, no normalization
            "size": len(actual_pcs),
            "original_voicing": original_voicing,  # MIDI note numbers
        }

    return CHORD_DICTIONARY


def load_bassflute_samples() -> Dict[str, Dict]:
    """Load bass flute samples metadata."""
    global BASSFLUTE_SAMPLES
    if BASSFLUTE_SAMPLES:
        return BASSFLUTE_SAMPLES

    if not BASSFLUTE_MANIFEST_PATH.exists():
        print(f"Warning: Bass flute manifest not found at {BASSFLUTE_MANIFEST_PATH}")
        return {}

    with open(BASSFLUTE_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    # Note name to pitch class mapping
    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = BASSFLUTE_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                # Get duration from audio file
                try:
                    audio = AudioSegment.from_file(audio_path)
                    duration_ms = len(audio)
                except Exception:
                    duration_ms = 20000  # Default 20 seconds if can't read
                BASSFLUTE_SAMPLES[sample_name] = {
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                    "duration_ms": duration_ms,
                }

    return BASSFLUTE_SAMPLES


def load_scelsipezzi_samples() -> Dict[str, Dict]:
    """Load Scelsi Pezzi samples metadata - sustaining single-pitch phrases."""
    global SCELSIPEZZI_SAMPLES
    if SCELSIPEZZI_SAMPLES:
        return SCELSIPEZZI_SAMPLES

    if not SCELSIPEZZI_MANIFEST_PATH.exists():
        print(f"Warning: Scelsi Pezzi manifest not found at {SCELSIPEZZI_MANIFEST_PATH}")
        return {}

    with open(SCELSIPEZZI_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    # Note name to pitch class mapping
    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = SCELSIPEZZI_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                try:
                    audio = AudioSegment.from_file(audio_path)
                    duration_ms = len(audio)
                except Exception:
                    duration_ms = 15000  # Default 15 seconds if can't read
                SCELSIPEZZI_SAMPLES[sample_name] = {
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                    "duration_ms": duration_ms,
                }

    return SCELSIPEZZI_SAMPLES


def load_brodero_samples() -> List[Dict]:
    """
    Load Brodero samples metadata.
    Returns a sorted list (alphanumeric order) for sequential playback.
    """
    global BRODERO_SAMPLES
    if BRODERO_SAMPLES:
        return list(BRODERO_SAMPLES.values())

    if not BRODERO_MANIFEST_PATH.exists():
        print(f"Warning: Brodero manifest not found at {BRODERO_MANIFEST_PATH}")
        return []

    with open(BRODERO_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    # Note name to pitch class mapping
    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            # Brodero samples are directly in the folder, not in a samples/ subfolder
            audio_path = BRODERO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                BRODERO_SAMPLES[sample_name] = {
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                }

    # Return sorted by name (alphanumeric order)
    sorted_samples = sorted(BRODERO_SAMPLES.values(), key=lambda x: x["name"])
    return sorted_samples


def load_organetta_samples() -> List[Dict]:
    """
    Load Organetta samples metadata.
    Returns a sorted list (alphanumeric order) for sequential playback.
    """
    global ORGANETTA_SAMPLES
    if ORGANETTA_SAMPLES:
        return list(ORGANETTA_SAMPLES.values())

    if not ORGANETTA_MANIFEST_PATH.exists():
        print(f"Warning: Organetta manifest not found at {ORGANETTA_MANIFEST_PATH}")
        return []

    with open(ORGANETTA_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    # Note name to pitch class mapping
    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            # Organetta samples are in the samples/ subfolder
            audio_path = ORGANETTA_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                ORGANETTA_SAMPLES[sample_name] = {
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                }

    # Return sorted by name (alphanumeric order)
    sorted_samples = sorted(ORGANETTA_SAMPLES.values(), key=lambda x: x["name"])
    return sorted_samples


def load_minorchordbeat_samples() -> List[Dict]:
    """
    Load MinorChordBeat samples metadata.
    Returns a sorted list (alphanumeric order) for sequential playback.
    """
    global MINORCHORDBEAT_SAMPLES
    if MINORCHORDBEAT_SAMPLES:
        return list(MINORCHORDBEAT_SAMPLES.values())

    if not MINORCHORDBEAT_MANIFEST_PATH.exists():
        print(f"Warning: MinorChordBeat manifest not found at {MINORCHORDBEAT_MANIFEST_PATH}")
        return []

    with open(MINORCHORDBEAT_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = MINORCHORDBEAT_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                MINORCHORDBEAT_SAMPLES[sample_name] = {
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                }

    sorted_samples = sorted(MINORCHORDBEAT_SAMPLES.values(), key=lambda x: x["name"])
    return sorted_samples


def load_mutebowl_samples() -> List[Dict]:
    """
    Load MuteBowl samples metadata.
    Returns a sorted list (alphanumeric order) for sequential playback.
    """
    global MUTEBOWL_SAMPLES
    if MUTEBOWL_SAMPLES:
        return list(MUTEBOWL_SAMPLES.values())

    if not MUTEBOWL_MANIFEST_PATH.exists():
        print(f"Warning: MuteBowl manifest not found at {MUTEBOWL_MANIFEST_PATH}")
        return []

    with open(MUTEBOWL_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = MUTEBOWL_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                MUTEBOWL_SAMPLES[sample_name] = {
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                }

    sorted_samples = sorted(MUTEBOWL_SAMPLES.values(), key=lambda x: x["name"])
    return sorted_samples


def load_prophetfalse_samples() -> List[Dict]:
    """
    Load Prophet False samples metadata.
    These are single-note synth samples.
    Returns a sorted list (alphanumeric order) for sequential playback.
    """
    global PROPHETFALSE_SAMPLES
    if PROPHETFALSE_SAMPLES:
        return PROPHETFALSE_SAMPLES

    if not PROPHETFALSE_MANIFEST_PATH.exists():
        print(f"Warning: Prophet False manifest not found at {PROPHETFALSE_MANIFEST_PATH}")
        return []

    with open(PROPHETFALSE_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    samples_list = []
    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = PROPHETFALSE_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                samples_list.append({
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                })

    PROPHETFALSE_SAMPLES = sorted(samples_list, key=lambda x: x["name"])
    return PROPHETFALSE_SAMPLES


def load_harmonicker_samples() -> List[Dict]:
    """
    Load Harmonicker samples metadata.
    These are harmonica chord/interval samples.
    Returns a sorted list (alphanumeric order) for sequential playback.
    """
    global HARMONICKER_SAMPLES
    if HARMONICKER_SAMPLES:
        return HARMONICKER_SAMPLES

    if not HARMONICKER_MANIFEST_PATH.exists():
        print(f"Warning: Harmonicker manifest not found at {HARMONICKER_MANIFEST_PATH}")
        return []

    with open(HARMONICKER_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    samples_list = []
    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = HARMONICKER_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                samples_list.append({
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                })

    HARMONICKER_SAMPLES = sorted(samples_list, key=lambda x: x["name"])
    return HARMONICKER_SAMPLES


def load_gothicharp_samples() -> List[Dict]:
    """
    Load Gothic Harp samples metadata.
    These are single-note harp samples for 16th note clouds.
    Returns a sorted list (alphanumeric order) for sequential playback.
    """
    global GOTHICHARP_SAMPLES
    if GOTHICHARP_SAMPLES:
        return GOTHICHARP_SAMPLES

    if not GOTHICHARP_MANIFEST_PATH.exists():
        print(f"Warning: Gothic Harp manifest not found at {GOTHICHARP_MANIFEST_PATH}")
        return []

    with open(GOTHICHARP_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    samples_list = []
    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = GOTHICHARP_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                samples_list.append({
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                })

    GOTHICHARP_SAMPLES = sorted(samples_list, key=lambda x: x["name"])
    return GOTHICHARP_SAMPLES


def load_gentleharpsi_samples() -> List[Dict]:
    """
    Load Gentle Harpsichord samples metadata.
    These are single-note harpsichord samples for 32nd note clouds.
    Returns a sorted list (alphanumeric order) for sequential playback.
    """
    global GENTLEHARPSI_SAMPLES
    if GENTLEHARPSI_SAMPLES:
        return GENTLEHARPSI_SAMPLES

    if not GENTLEHARPSI_MANIFEST_PATH.exists():
        print(f"Warning: Gentle Harpsichord manifest not found at {GENTLEHARPSI_MANIFEST_PATH}")
        return []

    with open(GENTLEHARPSI_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    samples_list = []
    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        # Handle pitch_classes directly if present
        if "pitch_classes" in sample_data:
            pitch_classes = sample_data["pitch_classes"]
        else:
            note_names = sample_data.get("note_names", [])
            pitch_classes = []
            for note in note_names:
                note_lower = note.lower()
                if note_lower in note_to_pc:
                    pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = GENTLEHARPSI_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                samples_list.append({
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "audio_path": audio_path,
                })

    GENTLEHARPSI_SAMPLES = sorted(samples_list, key=lambda x: x["name"])
    return GENTLEHARPSI_SAMPLES


def load_feedback_samples() -> List[Dict]:
    """
    Load Feedback samples metadata.
    These are loop samples that can overlap with random selection.
    """
    global FEEDBACK_SAMPLES
    if FEEDBACK_SAMPLES:
        return FEEDBACK_SAMPLES

    if not FEEDBACK_MANIFEST_PATH.exists():
        print(f"Warning: Feedback manifest not found at {FEEDBACK_MANIFEST_PATH}")
        return []

    with open(FEEDBACK_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    samples_list = []
    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = FEEDBACK_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                samples_list.append({
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                })

    FEEDBACK_SAMPLES = samples_list  # Don't sort - will be randomized during playback
    return FEEDBACK_SAMPLES


def load_stylo_samples() -> List[Dict]:
    """
    Load Stylo samples metadata.
    Similar to Organetta but plays twice as fast (every 2 seconds).
    """
    global STYLO_SAMPLES
    if STYLO_SAMPLES:
        return STYLO_SAMPLES

    if not STYLO_MANIFEST_PATH.exists():
        print(f"Warning: Stylo manifest not found at {STYLO_MANIFEST_PATH}")
        return []

    with open(STYLO_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    samples_list = []
    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = STYLO_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                samples_list.append({
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                })

    STYLO_SAMPLES = sorted(samples_list, key=lambda x: x["name"])
    return STYLO_SAMPLES


def load_trichords_samples() -> List[Dict]:
    """
    Load Trichords samples metadata.
    Continuous layer like bass flute - plays samples back-to-back.
    """
    global TRICHORDS_SAMPLES
    if TRICHORDS_SAMPLES:
        return TRICHORDS_SAMPLES

    if not TRICHORDS_MANIFEST_PATH.exists():
        print(f"Warning: Trichords manifest not found at {TRICHORDS_MANIFEST_PATH}")
        return []

    with open(TRICHORDS_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    samples_list = []
    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = TRICHORDS_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                samples_list.append({
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                })

    TRICHORDS_SAMPLES = sorted(samples_list, key=lambda x: x["name"])
    return TRICHORDS_SAMPLES


def load_tremolo_oct_samples() -> List[Dict]:
    """
    Load Tremolo Oct samples metadata.
    Interval layer like organetta - plays at fixed intervals.
    """
    global TREMOLO_OCT_SAMPLES
    if TREMOLO_OCT_SAMPLES:
        return TREMOLO_OCT_SAMPLES

    if not TREMOLO_OCT_MANIFEST_PATH.exists():
        print(f"Warning: Tremolo Oct manifest not found at {TREMOLO_OCT_MANIFEST_PATH}")
        return []

    with open(TREMOLO_OCT_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    samples_list = []
    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = TREMOLO_OCT_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                samples_list.append({
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                })

    TREMOLO_OCT_SAMPLES = sorted(samples_list, key=lambda x: x["name"])
    return TREMOLO_OCT_SAMPLES


def load_averyviolin_samples() -> Dict[str, Dict]:
    """Load Avery Violin Phrase samples metadata (continuous, like bass flute)."""
    global AVERYVIOLIN_SAMPLES
    if AVERYVIOLIN_SAMPLES:
        return AVERYVIOLIN_SAMPLES

    if not AVERYVIOLIN_MANIFEST_PATH.exists():
        print(f"Warning: Avery Violin manifest not found at {AVERYVIOLIN_MANIFEST_PATH}")
        return {}

    with open(AVERYVIOLIN_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = AVERYVIOLIN_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                try:
                    audio = AudioSegment.from_file(audio_path)
                    duration_ms = len(audio)
                except Exception:
                    duration_ms = 20000
                AVERYVIOLIN_SAMPLES[sample_name] = {
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                    "duration_ms": duration_ms,
                }

    return AVERYVIOLIN_SAMPLES


def load_dictamel_samples() -> Dict[str, Dict]:
    """Load Dictamel samples metadata (continuous, like bass flute)."""
    global DICTAMEL_SAMPLES
    if DICTAMEL_SAMPLES:
        return DICTAMEL_SAMPLES

    if not DICTAMEL_MANIFEST_PATH.exists():
        print(f"Warning: Dictamel manifest not found at {DICTAMEL_MANIFEST_PATH}")
        return {}

    with open(DICTAMEL_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = DICTAMEL_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                try:
                    audio = AudioSegment.from_file(audio_path)
                    duration_ms = len(audio)
                except Exception:
                    duration_ms = 20000
                DICTAMEL_SAMPLES[sample_name] = {
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                    "duration_ms": duration_ms,
                }

    return DICTAMEL_SAMPLES


def load_laken_samples() -> List[Dict]:
    """
    Load Laken samples metadata.
    These are single-note samples similar to gothic harp.
    Returns a sorted list (alphanumeric order) for sequential playback.
    """
    global LAKEN_SAMPLES
    if LAKEN_SAMPLES:
        return LAKEN_SAMPLES

    if not LAKEN_MANIFEST_PATH.exists():
        print(f"Warning: Laken manifest not found at {LAKEN_MANIFEST_PATH}")
        return []

    with open(LAKEN_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    samples_list = []
    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = LAKEN_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                samples_list.append({
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                })

    LAKEN_SAMPLES = sorted(samples_list, key=lambda x: x["name"])
    return LAKEN_SAMPLES


def load_minorchords_samples() -> List[Dict]:
    """
    Load minor chord samples metadata.
    Returns a sorted list (alphanumeric order) for sequential playback.
    """
    global MINORCHORDS_SAMPLES
    if MINORCHORDS_SAMPLES:
        return MINORCHORDS_SAMPLES

    if not MINORCHORDS_MANIFEST_PATH.exists():
        print(f"Warning: MinorChords manifest not found at {MINORCHORDS_MANIFEST_PATH}")
        return []

    with open(MINORCHORDS_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    samples_list = []
    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = MINORCHORDS_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                samples_list.append({
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                })

    MINORCHORDS_SAMPLES = sorted(samples_list, key=lambda x: x["name"])
    return MINORCHORDS_SAMPLES


def load_majorchords_samples() -> List[Dict]:
    """
    Load major chord samples metadata.
    Returns a sorted list (alphanumeric order) for sequential playback.
    """
    global MAJORCHORDS_SAMPLES
    if MAJORCHORDS_SAMPLES:
        return MAJORCHORDS_SAMPLES

    if not MAJORCHORDS_MANIFEST_PATH.exists():
        print(f"Warning: MajorChords manifest not found at {MAJORCHORDS_MANIFEST_PATH}")
        return []

    with open(MAJORCHORDS_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    samples_list = []
    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = MAJORCHORDS_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                samples_list.append({
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                })

    MAJORCHORDS_SAMPLES = sorted(samples_list, key=lambda x: x["name"])
    return MAJORCHORDS_SAMPLES


def load_progression_samples(
    manifest_path: Path,
    audio_dir: Path,
    global_list: List[Dict],
    library_name: str
) -> List[Dict]:
    """
    Load progression samples metadata (multi-chord sequences with exact timing).

    These samples have chord_sequence with start_time for each chord.
    Only loads samples that have valid chord_sequence data (ignoring broken ones).

    Returns a sorted list of sample dicts with:
    - name: sample name
    - audio_path: path to audio file
    - duration: total duration in seconds
    - chord_sequence: list of {start_time, pitch_classes, chord_name (optional)}
    - first_chord: {pitch_classes, start_time}
    - last_chord: {pitch_classes, start_time}
    """
    if global_list:
        return global_list

    if not manifest_path.exists():
        print(f"Warning: {library_name} manifest not found at {manifest_path}")
        return []

    with open(manifest_path) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'db': 1, 'd': 2, 'ds': 3, 'ef': 3, 'eb': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'gb': 6, 'g': 7, 'gs': 8, 'af': 8, 'ab': 8, 'a': 9,
        'as': 10, 'bf': 10, 'bb': 10, 'b': 11
    }

    samples_list = []
    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        chord_sequence = sample_data.get("chord_sequence", [])
        duration = sample_data.get("duration", 0)

        # Skip samples without chord data
        if not chord_sequence or len(chord_sequence) < 2:
            continue

        # Convert chord_sequence to use pitch_classes as sets
        processed_sequence = []
        for chord in chord_sequence:
            start_time = chord.get("start_time", 0)

            # Get pitch classes - either directly or from note_names
            if "pitch_classes" in chord:
                pcs = set(chord["pitch_classes"])
            elif "note_names" in chord:
                pcs = set()
                for note in chord["note_names"]:
                    note_lower = note.lower()
                    if note_lower in note_to_pc:
                        pcs.add(note_to_pc[note_lower])
            else:
                continue

            if pcs:
                processed_sequence.append({
                    "start_time": start_time,
                    "pitch_classes": pcs,
                    "chord_name": chord.get("chord_name", "")
                })

        if len(processed_sequence) < 2:
            continue

        audio_path = audio_dir / f"{sample_name}.wav"
        if not audio_path.exists():
            continue

        samples_list.append({
            "name": sample_name,
            "audio_path": audio_path,
            "duration": duration,
            "chord_sequence": processed_sequence,
            "first_chord": processed_sequence[0],
            "last_chord": processed_sequence[-1],
            "num_chords": len(processed_sequence),
        })

    samples_list = sorted(samples_list, key=lambda x: x["name"])
    global_list.extend(samples_list)
    return samples_list


def load_glaz_sax_samples() -> List[Dict]:
    """Load Glaz Sax Chorale progression samples."""
    global GLAZ_SAX_SAMPLES
    return load_progression_samples(
        GLAZ_SAX_MANIFEST_PATH, GLAZ_SAX_AUDIO_DIR,
        GLAZ_SAX_SAMPLES, "GlazSaxChorales"
    )


def load_hyacinthe_samples() -> List[Dict]:
    """Load Hyacinthe progression samples."""
    global HYACINTHE_SAMPLES
    return load_progression_samples(
        HYACINTHE_MANIFEST_PATH, HYACINTHE_AUDIO_DIR,
        HYACINTHE_SAMPLES, "Hyacinthe"
    )


def load_kraus_samples() -> List[Dict]:
    """Load Kraus Chorale progression samples."""
    global KRAUS_SAMPLES
    return load_progression_samples(
        KRAUS_MANIFEST_PATH, KRAUS_AUDIO_DIR,
        KRAUS_SAMPLES, "KrausChorale"
    )


def load_godette_samples() -> List[Dict]:
    """
    Load Godette samples metadata.
    Chord-triggered layer: plays on chord changes if a fitting sample exists.
    Only one sample can play at a time.
    """
    global GODETTE_SAMPLES
    if GODETTE_SAMPLES:
        return GODETTE_SAMPLES

    if not GODETTE_MANIFEST_PATH.exists():
        print(f"Warning: Godette manifest not found at {GODETTE_MANIFEST_PATH}")
        return []

    with open(GODETTE_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    samples_list = []
    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = GODETTE_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                samples_list.append({
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                })

    GODETTE_SAMPLES = sorted(samples_list, key=lambda x: x["name"])
    print(f"  Loaded {len(GODETTE_SAMPLES)} Godette samples")
    return GODETTE_SAMPLES


def get_chord_quality(chord_name: str) -> str:
    """
    Determine if a chord is major, minor, or other based on chord_type.
    Returns 'major', 'minor', or 'other'.
    """
    chord_dict = load_chord_dictionary()
    chord_data = chord_dict.get(chord_name, {})
    chord_type = chord_data.get("chord_type", "")

    # chord_type starts with _m = minor, _M = major
    if chord_type.startswith("_m"):
        return "minor"
    elif chord_type.startswith("_M"):
        return "major"
    else:
        # Dominant, diminished, augmented, etc. - treat as "other"
        return "other"


def fits_in_chord(sample_pcs: Set[int], chord_pcs: Set[int], transposition: int) -> bool:
    """
    Check if sample pitch classes fit within chord when transposed.
    SNAPS-style: all transposed sample PCs must be in target chord.
    """
    transposed = {(pc + transposition) % 12 for pc in sample_pcs}
    return transposed.issubset(chord_pcs)


def can_sample_fit_chord(sample_pcs: Set[int], chord_pcs: Set[int]) -> bool:
    """
    Check if a sample can fit a chord at ANY valid transposition.
    Returns True if at least one valid transposition exists.
    """
    for t in range(MIN_TRANSPOSITION, MAX_TRANSPOSITION + 1):
        if fits_in_chord(sample_pcs, chord_pcs, t):
            return True
    return False


def find_next_fitting_oneshot(
    samples: List[Dict],
    start_idx: int,
    chord_pcs: Set[int]
) -> Optional[Tuple[int, Dict]]:
    """
    Find the next oneshot sample (starting from start_idx) that can fit the chord.

    Returns (sample_index, sample_data) or None if no sample fits.
    Wraps around the sample list once.
    """
    n = len(samples)
    for offset in range(n):
        idx = (start_idx + offset) % n
        sample = samples[idx]
        if can_sample_fit_chord(sample["pitch_classes"], chord_pcs):
            return (idx, sample)
    return None


def get_harmonic_event_at_time(
    time_ms: float,
    harmonic_timeline: List['HarmonicEvent']
) -> Optional['HarmonicEvent']:
    """
    Find the harmonic event active at a given time.
    """
    for he in harmonic_timeline:
        if he.start_ms <= time_ms < he.end_ms:
            return he
    return None


def find_bassflute_transposition(
    sample_pcs: Set[int],
    chord_pcs: Set[int],
    chord_root: int = 0
) -> Optional[int]:
    """
    Find optimal transposition for a bass flute sample to fit in the chord.

    Uses SNAPS "elastic pull" algorithm:
    - Biases toward 0 (no transposition) to prevent drift
    - Biases downward when equidistant (sounds more natural)

    Returns semitones to transpose, or None if no valid transposition.
    """
    valid = []
    for t in range(MIN_TRANSPOSITION, MAX_TRANSPOSITION + 1):
        if fits_in_chord(sample_pcs, chord_pcs, t):
            valid.append(t)

    if not valid:
        return None

    # Score: |distance_from_home| + upward_penalty
    def score(t):
        return abs(t) + (0.5 if t > 0 else 0.0)

    # Pick lowest score (ties favor negative/lower)
    best = valid[0]
    best_score = score(best)

    for t in valid:
        s = score(t)
        if s < best_score or (s == best_score and t < best):
            best = t
            best_score = s

    return best


def apply_glissando(audio: AudioSegment, semitones_start: float, semitones_end: float, gliss_ms: int = GLISSANDO_MS) -> AudioSegment:
    """
    Apply a pitch glissando using vectorized varispeed.

    The audio starts at semitones_start and glides to semitones_end over gliss_ms.
    Uses quarter-sine easing and np.interp for smooth tape-style bend.

    Vectorized implementation: builds rate curve, uses cumsum for input positions,
    then np.interp for resampling. No Python loops in the inner processing.
    """
    if abs(semitones_start - semitones_end) < 0.01:
        # No glissando needed
        return audio

    original_rate = audio.frame_rate
    channels = audio.channels
    sample_width = audio.sample_width

    # Convert to numpy array
    raw_samples = audio.get_array_of_samples()
    if len(raw_samples) == 0:
        return audio

    samples = np.array(raw_samples, dtype=np.float64)

    # Handle stereo by reshaping
    if channels == 2:
        # Ensure even length for stereo
        if len(samples) % 2 != 0:
            samples = samples[:-1]
        samples = samples.reshape((-1, 2))
    else:
        samples = samples.reshape((-1, 1))

    num_input_samples = len(samples)
    if num_input_samples < 2:
        return audio

    gliss_samples = int(gliss_ms * original_rate / 1000)
    gliss_samples = min(gliss_samples, num_input_samples - 1)
    if gliss_samples < 1:
        return audio

    # VECTORIZED GLISSANDO PORTION
    # Estimate output length for glissando
    avg_rate = (2 ** (semitones_start / 12.0) + 2 ** (semitones_end / 12.0)) / 2
    if avg_rate <= 0:
        avg_rate = 1.0
    output_gliss_samples = max(1, int(gliss_samples / avg_rate))

    # Build progress array (0 to 1) for output samples
    progress = np.linspace(0, 1, output_gliss_samples)

    # Quarter-sine easing (vectorized)
    eased = np.sin(progress * np.pi / 2)

    # Semitone curve (vectorized)
    semitone_curve = semitones_start + (semitones_end - semitones_start) * eased

    # Rate curve (vectorized) - playback rate at each output sample
    rate_curve = 2 ** (semitone_curve / 12.0)

    # Cumulative sum gives input positions for each output sample
    # Start at 0, then accumulate rates
    input_positions = np.concatenate([[0], np.cumsum(rate_curve[:-1])])

    # Clamp positions to valid range
    input_positions = np.clip(input_positions, 0, num_input_samples - 1.001)

    # Resample each channel using np.interp (vectorized)
    input_indices = np.arange(num_input_samples)
    output_gliss = np.zeros((output_gliss_samples, channels), dtype=np.float64)

    for ch in range(channels):
        output_gliss[:, ch] = np.interp(input_positions, input_indices, samples[:, ch])

    # Figure out where we ended up in the input
    final_input_pos = input_positions[-1] + rate_curve[-1] if output_gliss_samples > 0 else 0
    rest_start_input = int(final_input_pos)

    output_parts = [output_gliss]

    # VECTORIZED REST PORTION (after glissando)
    if rest_start_input < num_input_samples:
        rest_samples = samples[rest_start_input:]
        num_rest_samples = len(rest_samples)

        if num_rest_samples > 0:
            if abs(semitones_end) > 0.01:
                # Apply final transposition (constant rate)
                final_rate = 2 ** (semitones_end / 12.0)
                rest_output_len = max(1, int(num_rest_samples / final_rate))

                # Vectorized resampling for rest
                rest_output_positions = np.arange(rest_output_len) * final_rate
                rest_output_positions = np.clip(rest_output_positions, 0, num_rest_samples - 1.001)
                rest_input_indices = np.arange(num_rest_samples)

                output_rest = np.zeros((rest_output_len, channels), dtype=np.float64)
                for ch in range(channels):
                    output_rest[:, ch] = np.interp(rest_output_positions, rest_input_indices, rest_samples[:, ch])

                output_parts.append(output_rest)
            else:
                # No transposition needed for rest
                output_parts.append(rest_samples)

    # Combine all parts
    output = np.vstack(output_parts)

    # Flatten back for pydub (interleaved stereo)
    output = output.flatten()

    # Ensure even length for stereo
    if channels == 2 and len(output) % 2 != 0:
        output = output[:-1]

    # Ensure non-empty
    if len(output) == 0:
        return audio

    # Clip and convert back to int16
    output = np.clip(output, -32768, 32767).astype(np.int16)

    # Create new AudioSegment with error handling
    try:
        return AudioSegment(
            data=output.tobytes(),
            sample_width=sample_width,
            frame_rate=original_rate,
            channels=channels
        )
    except Exception:
        # If byte alignment fails, fall back to original audio
        return audio


@dataclass
class BassFluteEvent:
    """A bass flute one-shot layered over the chain."""
    sample_name: str
    start_ms: float
    original_pcs: Set[int]
    # Note: transposition is now calculated dynamically based on harmonic state at each moment


@dataclass
class HarmonicEvent:
    """A chord/harmonic state at a specific time in the chain."""
    start_ms: float
    end_ms: float
    chord_pcs: Set[int]
    chord_root: int
    chord_name: str


@dataclass
class RenderedSkeletonEvent:
    """
    Tracks actual rendered timing for a skeleton sample.

    This is the single source of truth for timing - audio render creates
    the clock, MIDI follows that clock. Not the other way around.
    """
    sample_name: str
    transposition: int
    source_duration_ms: float        # Original duration from samplesData.json
    rendered_start_ms: float         # Actual start time in rendered audio
    rendered_duration_ms: float      # Actual duration after varispeed
    rendered_end_ms: float           # = rendered_start_ms + rendered_duration_ms
    first_pcs: Set[int]              # Pitch classes at start (transposed)
    last_pcs: Set[int]               # Pitch classes at end (transposed)
    inferred_chord: Dict             # The bridging chord (chord B)
    chord_b_onset_ratio: float       # When chord B starts as ratio of duration
    # Derived timing in rendered space
    chord_a_start_ms: float = 0      # = rendered_start_ms
    chord_b_start_ms: float = 0      # = rendered_start_ms + (rendered_duration_ms * chord_b_onset_ratio)


def has_semitone_cluster(pcs: Set[int], min_consecutive: int = 3) -> bool:
    """
    Check if a pitch class set contains a semitone cluster.

    A semitone cluster is defined as min_consecutive or more consecutive
    semitones (e.g., C, C#, D = {0, 1, 2}).

    Returns True if cluster found, False otherwise.
    """
    if len(pcs) < min_consecutive:
        return False

    sorted_pcs = sorted(pcs)

    # Check all possible starting positions in the chromatic circle
    for start_pc in range(12):
        consecutive = 0
        for offset in range(12):
            pc = (start_pc + offset) % 12
            if pc in pcs:
                consecutive += 1
                if consecutive >= min_consecutive:
                    return True
            else:
                consecutive = 0

    return False


def pitch_class_to_note(pc: int) -> str:
    return NOTE_NAMES[pc % 12]


def normalize_audio_peak(audio: AudioSegment, target_db: float = SAMPLE_NORMALIZE_DB) -> AudioSegment:
    """
    Peak normalize audio to a target dB level.

    This is NOT compression - it simply scales the entire audio so that
    the peak reaches the target level. Dynamics are preserved.

    Args:
        audio: Input AudioSegment
        target_db: Target peak level in dB (e.g., -5.0)

    Returns:
        Normalized AudioSegment
    """
    # Get current peak level
    current_peak = audio.max_dBFS
    if current_peak == float('-inf'):
        return audio  # Silent audio, can't normalize

    # Calculate gain needed to reach target
    gain_db = target_db - current_peak

    # Apply gain
    return audio + gain_db


def transpose_set(pcs: Set[int], semitones: int) -> Set[int]:
    """Transpose a set of pitch classes."""
    return {(pc + semitones) % 12 for pc in pcs}


def score_chord_quality(chord_name: str, chord_type: str) -> float:
    """
    Score a chord based on quality. Lower score = more preferred.

    Preferences (strongly weighted toward major/minor):
    - Major and minor chords: best (score 0)
    - Dominant 7ths, sus chords: okay (score 5)
    - Augmented: not great (score 8)
    - Diminished and half-diminished: worst (score 10)
    """
    name_lower = chord_name.lower()
    type_lower = chord_type.lower() if chord_type else ""

    # Check for diminished FIRST (worst) - before minor check
    if 'dim' in name_lower or 'dim' in type_lower or '°' in chord_name:
        return 10.0

    # Check for half-diminished BEFORE minor check
    # m7♭5, mø, half-dim all get penalized
    if 'ø' in chord_name or '♭5' in chord_name or 'b5' in name_lower or 'half' in type_lower:
        return 10.0

    # Check for augmented
    if 'aug' in name_lower or '+' in chord_name or '#5' in chord_name:
        return 8.0

    # Check for major (best)
    if 'M' in chord_name or 'maj' in type_lower:
        return 0.0

    # Check for minor (best) - but not if it has ♭5
    if '_m' in chord_name or 'min' in type_lower:
        return 0.0

    # Dominant 7ths, sus, etc (okay but not preferred)
    if '7' in chord_name or 'sus' in name_lower or 'dom' in type_lower:
        return 5.0

    # Default: slightly penalized
    return 6.0


def score_root_movement(prev_root: Optional[int], new_root: int) -> float:
    """
    Score root movement. Lower score = more preferred.

    Preferences:
    - Fifth (7 semitones up/down): best (score 0)
    - Tritone (6 semitones): very good (score 0.5)
    - Fourth (5 semitones): good (score 1)
    - Other: neutral (score 2)
    """
    if prev_root is None:
        return 0  # First chord, no preference

    # Calculate interval (always positive, 0-6)
    interval = abs(new_root - prev_root)
    if interval > 6:
        interval = 12 - interval

    if interval == 5:  # Fifth (down) or fourth (up)
        return 0.0
    elif interval == 7 % 12:  # Fifth (up) - same as 5 due to inversion
        return 0.0
    elif interval == 6:  # Tritone
        return 0.5
    elif interval == 4:  # Major third
        return 1.5
    elif interval == 3:  # Minor third
        return 1.5
    else:
        return 2.0


def score_voice_leading(prev_chord_pcs: Optional[Set[int]], new_chord_pcs: Set[int]) -> float:
    """
    Score voice leading based on pitch class similarity.
    Lower score = smoother voice leading (more shared notes).

    The score is based on how many pitch classes are NOT shared.
    Maximum similarity = all notes shared = score 0.
    """
    if prev_chord_pcs is None:
        return 0  # First chord, no preference

    # Count shared pitch classes
    shared = len(prev_chord_pcs & new_chord_pcs)
    total_unique = len(prev_chord_pcs | new_chord_pcs)

    # Score based on how many notes changed
    # More shared = lower score = better
    if total_unique == 0:
        return 0

    # Percentage of notes that are NOT shared (0 = perfect, 1 = no overlap)
    change_ratio = 1.0 - (shared / total_unique)

    # Scale to 0-5 range for weighting
    return change_ratio * 5.0


def infer_chord_supersets(
    collection: Set[int],
    max_results: int = 50,
    target_sizes: List[int] = [4, 5, 6],
    exclude_clusters: bool = True,
    prev_root: Optional[int] = None,
    prev_chord_pcs: Optional[Set[int]] = None
) -> List[Dict]:
    """
    Find chords that contain the given collection as a subset.

    Uses the rich chord vocabulary from chords_no_supersets.json.
    Optionally excludes chords containing semitone clusters.

    Scoring preferences (in order of importance):
    1. Voice leading - smooth transitions (max pitch class similarity)
    2. Quality - major and minor chords strongly preferred
    3. Named chords preferred over unnamed
    4. Root movement - fifths and tritones preferred
    5. Specificity - fewer added notes preferred

    Returns list of {pitch_classes, name, size, specificity, quality_score, root_score, voice_leading_score}.
    """
    chord_dict = load_chord_dictionary()
    results = []
    seen_pcs = set()

    # Phase 1: Search chord dictionary for supersets
    for chord_key, chord_data in chord_dict.items():
        chord_pcs = set(chord_data["pitch_classes"])

        # Skip if chord doesn't contain the collection
        if not collection.issubset(chord_pcs):
            continue

        # Skip if chord has semitone cluster (if exclusion enabled)
        if exclude_clusters and has_semitone_cluster(chord_pcs):
            continue

        # Skip if we've seen this pitch class set before
        pcs_key = tuple(sorted(chord_pcs))
        if pcs_key in seen_pcs:
            continue
        seen_pcs.add(pcs_key)

        chord_type = chord_data.get("chord_type", "")
        root = chord_data.get("root", 0)
        quality_score = score_chord_quality(chord_key, chord_type)
        root_score = score_root_movement(prev_root, root)
        voice_leading_score = score_voice_leading(prev_chord_pcs, chord_pcs)

        results.append({
            "pitch_classes": sorted(chord_pcs),
            "name": chord_key,
            "chord_type": chord_type,
            "root": root,
            "size": len(chord_pcs),
            "specificity": len(chord_pcs) - len(collection),
            "quality_score": quality_score,
            "root_score": root_score,
            "voice_leading_score": voice_leading_score,
            "named": True,
        })

    # Phase 2 REMOVED: All chords must come from the dictionary.
    # With 3-5 pitch class inputs, the dictionary always has sufficient coverage.

    # Sort: prioritize voice leading, then quality, then root, then specificity
    # Lower total score = better
    # All chords are named (from dictionary) - no unnamed chords allowed
    def total_score(x):
        return (
            x.get("voice_leading_score", 5.0),  # Voice leading FIRST (most important)
            x.get("quality_score", 8.0),  # Quality (major/minor strongly preferred)
            x.get("root_score", 2.0),  # Root movement
            x.get("specificity", 0),  # Fewer added notes preferred
            x["pitch_classes"][0],  # Tiebreaker
        )

    results.sort(key=total_score)

    return results[:max_results]


def find_fitting_sample(
    target_chord: Set[int],
    samples: Dict[str, Dict],
    used_counts: Dict[str, int],
    max_uses: int = 2,
    transposition_range: range = range(-6, 7),
    exclude_clusters: bool = True,
    prefer_alternate_library: Optional[str] = None
) -> Optional[Tuple[str, int, Set[int]]]:
    """
    Find a sample whose first_collection can be transposed to fit within target_chord.

    Samples with semitone clusters are already filtered out at the build_chain level,
    but this function operates on the pre-filtered valid_samples dict.

    Args:
        prefer_alternate_library: If "feldman", prefer Handel samples; if "handel", prefer Feldman.
                                  This encourages alternation between libraries.

    Returns (sample_name, transposition, transposed_first_collection) or None.
    """
    candidates = []

    for name, data in samples.items():
        if name.startswith("_"):
            continue
        if used_counts.get(name, 0) >= max_uses:
            continue

        first_chord = data.get("first_chord", {})
        first_pcs = set(first_chord.get("pitch_classes", []))

        if not first_pcs:
            continue

        # Determine library for alternation preference
        is_handel = name.startswith("handel_")

        # Try each transposition
        for trans in transposition_range:
            transposed = transpose_set(first_pcs, trans)

            # Check if transposed first_collection is subset of target chord
            if transposed.issubset(target_chord):
                # Alternation bonus: STRONGLY prefer samples from the other library
                alt_bonus = 0
                if prefer_alternate_library == "feldman" and is_handel:
                    alt_bonus = -100  # Very strong preference for Handel when coming from Feldman
                elif prefer_alternate_library == "handel" and not is_handel:
                    alt_bonus = -100  # Very strong preference for Feldman when coming from Handel

                candidates.append({
                    "name": name,
                    "transposition": trans,
                    "transposed_pcs": transposed,
                    "original_pcs": first_pcs,
                    "uses": used_counts.get(name, 0),
                    "alt_bonus": alt_bonus,
                    "is_handel": is_handel,
                })

    if not candidates:
        return None

    # Sort: alternation bonus first, then fewer uses, then smaller transposition
    candidates.sort(key=lambda x: (x["alt_bonus"], x["uses"], abs(x["transposition"])))

    best = candidates[0]
    return (best["name"], best["transposition"], best["transposed_pcs"])


@dataclass
class ChainLink:
    sample: str
    transposition: int
    first_pcs: Set[int]
    last_pcs: Set[int]
    inferred_chord: Dict
    duration_ms: float
    chord_b_onset_ratio: float = 0.5  # When chord B starts (as ratio of duration)
    audio_dir: Path = None  # Directory containing the audio file
    chord_sequence: List[Dict] = None  # For progression samples: list of {start_time, pitch_classes}


def build_chain(
    samples: Dict[str, Dict],
    audio_dirs: Dict[str, Path] = None,
    start_sample: Optional[str] = None,
    max_uses: int = 2,
    seed: Optional[int] = None,
    verbose: bool = True,
    exclude_clusters: bool = True
) -> List[ChainLink]:
    """
    Build a chain using Quadruple Hierarchy logic.

    Args:
        exclude_clusters: If True, disqualify samples with semitone clusters
                         (3+ consecutive semitones) in first or last chord.
    """
    if seed is not None:
        random.seed(seed)

    # Load chord dictionary for inference
    load_chord_dictionary()

    # Filter to valid samples
    valid_samples = {}
    disqualified = []

    for k, v in samples.items():
        if k.startswith("_"):
            continue
        if "first_chord" not in v or "last_chord" not in v:
            continue

        first_pcs = set(v["first_chord"].get("pitch_classes", []))
        last_pcs = set(v["last_chord"].get("pitch_classes", []))

        # Check for semitone clusters
        if exclude_clusters:
            if has_semitone_cluster(first_pcs) or has_semitone_cluster(last_pcs):
                disqualified.append(k)
                continue

        valid_samples[k] = v

    if verbose and disqualified:
        print(f"Disqualified {len(disqualified)} samples with semitone clusters:")
        for name in disqualified[:5]:
            sample = samples[name]
            first = sample["first_chord"].get("pitch_classes", [])
            last = sample["last_chord"].get("pitch_classes", [])
            first_notes = [NOTE_NAMES[pc] for pc in first]
            last_notes = [NOTE_NAMES[pc] for pc in last]
            print(f"  {name}: first={first_notes}, last={last_notes}")
        if len(disqualified) > 5:
            print(f"  ...and {len(disqualified) - 5} more")

    if not valid_samples:
        raise ValueError("No valid samples found")

    # Track usage
    used_counts: Dict[str, int] = {}

    # Pick starting sample
    if start_sample and start_sample in valid_samples:
        current_name = start_sample
    else:
        current_name = random.choice(list(valid_samples.keys()))

    chain: List[ChainLink] = []
    prev_root: Optional[int] = None  # Track previous chord root for movement scoring
    prev_chord_pcs: Optional[Set[int]] = None  # Track previous chord for voice leading

    while True:
        sample_data = valid_samples[current_name]

        # Get first and last collections
        first_pcs = set(sample_data["first_chord"]["pitch_classes"])
        last_pcs = set(sample_data["last_chord"]["pitch_classes"])

        # Get duration and chord B onset ratio (from onset detection)
        duration_ms = sample_data.get("duration_ms", 4000)
        chord_b_onset_ratio = sample_data.get("chord_b_onset_ratio", 0.5)

        # Current transposition (0 for first, or from previous search)
        current_trans = 0 if not chain else chain[-1].inferred_chord.get("next_trans", 0)

        # Apply transposition to this sample's collections
        if current_trans != 0:
            first_pcs = transpose_set(first_pcs, current_trans)
            last_pcs = transpose_set(last_pcs, current_trans)

        # Infer chord supersets from last_collection
        # Pass prev_root and prev_chord_pcs for voice leading and root movement scoring
        chord_candidates = infer_chord_supersets(
            last_pcs,
            max_results=50,
            target_sizes=[4, 5, 6],
            prev_root=prev_root,
            prev_chord_pcs=prev_chord_pcs
        )

        if not chord_candidates:
            if verbose:
                print(f"  No chord supersets found for {last_pcs}, ending chain")
            break

        if verbose:
            last_notes = [pitch_class_to_note(pc) for pc in sorted(last_pcs)]
            print(f"\n{len(chain)+1}. {current_name}" + (f" (trans {current_trans:+d})" if current_trans else ""))
            print(f"   First: {[pitch_class_to_note(pc) for pc in sorted(first_pcs)]}")
            print(f"   Last:  {last_notes}")

        # Record usage
        used_counts[current_name] = used_counts.get(current_name, 0) + 1

        # Try each chord candidate until we find one with a fitting sample
        chosen_chord = None
        result = None

        # Determine current library for alternation
        current_library = "handel" if current_name.startswith("handel_") else "feldman"

        # First pass: try to find a sample from the ALTERNATE library only
        alternate_only_samples = {
            k: v for k, v in valid_samples.items()
            if (current_library == "feldman" and k.startswith("handel_")) or
               (current_library == "handel" and not k.startswith("handel_"))
        }

        for candidate_chord in chord_candidates:
            candidate_pcs = set(candidate_chord["pitch_classes"])

            # First try alternate library only
            result = find_fitting_sample(
                candidate_pcs,
                alternate_only_samples,
                used_counts,
                max_uses=max_uses,
            )

            if result is not None:
                chosen_chord = candidate_chord
                break

        # Second pass: if no alternate found, try any library
        if result is None:
            for candidate_chord in chord_candidates:
                candidate_pcs = set(candidate_chord["pitch_classes"])

                result = find_fitting_sample(
                    candidate_pcs,
                    valid_samples,
                    used_counts,
                    max_uses=max_uses,
                )

                if result is not None:
                    chosen_chord = candidate_chord
                    break

        if chosen_chord is None or result is None:
            # No chord worked - use first candidate for logging, end chain
            chosen_chord = chord_candidates[0]
            if verbose:
                chord_notes = [pitch_class_to_note(pc) for pc in chosen_chord["pitch_classes"]]
                print(f"   → Tried {len(chord_candidates)} chords, none had fitting samples")
                print(f"   → Best attempt: {chosen_chord['name']} {chord_notes}")
                print(f"   → Ending chain")
            chain.append(ChainLink(
                sample=current_name,
                transposition=current_trans,
                first_pcs=first_pcs,
                last_pcs=last_pcs,
                inferred_chord=chosen_chord,
                duration_ms=duration_ms,
                chord_b_onset_ratio=chord_b_onset_ratio,
                audio_dir=audio_dirs.get(current_name, AUDIO_DIR) if audio_dirs else AUDIO_DIR,
                chord_sequence=sample_data.get("chord_sequence"),
            ))
            break

        chosen_chord_pcs = set(chosen_chord["pitch_classes"])
        next_name, next_trans, next_first_pcs = result

        if verbose:
            chord_notes = [pitch_class_to_note(pc) for pc in chosen_chord["pitch_classes"]]
            next_first_notes = [pitch_class_to_note(pc) for pc in sorted(next_first_pcs)]
            print(f"   → Inferred chord: {chosen_chord['name']} {chord_notes}")
            print(f"   → Next: {next_name} (trans {next_trans:+d}), first={next_first_notes} ⊆ chord")

        # Store the transposition for next iteration
        chosen_chord["next_trans"] = next_trans

        chain.append(ChainLink(
            sample=current_name,
            transposition=current_trans,
            first_pcs=first_pcs,
            last_pcs=last_pcs,
            inferred_chord=chosen_chord,
            duration_ms=duration_ms,
            chord_b_onset_ratio=chord_b_onset_ratio,
            audio_dir=audio_dirs.get(current_name, AUDIO_DIR) if audio_dirs else AUDIO_DIR,
            chord_sequence=sample_data.get("chord_sequence"),
        ))

        # Update prev_root and prev_chord_pcs for next iteration
        prev_root = chosen_chord.get("root", 0)
        prev_chord_pcs = set(chosen_chord.get("pitch_classes", []))

        current_name = next_name

    return chain


def generate_chord_midi(
    voicing: List[int],
    root_pc: int,
    duration_sec: float,
    start_time_sec: float = 0.0
) -> 'pretty_midi.PrettyMIDI':
    """
    Generate a MIDI object with a chord voicing + root bass note.

    Args:
        voicing: MIDI note numbers from original_voicing
        root_pc: Root pitch class (0-11)
        duration_sec: How long the chord should sustain
        start_time_sec: When the chord starts

    Returns:
        PrettyMIDI object with the chord
    """
    import pretty_midi

    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    # Add root in bass register (octave 2)
    root_midi = 36 + root_pc  # C2 = 36
    bass_note = pretty_midi.Note(
        velocity=80,
        pitch=root_midi,
        start=start_time_sec,
        end=start_time_sec + duration_sec
    )
    piano.notes.append(bass_note)

    # Add voicing notes
    for midi_note in voicing:
        note = pretty_midi.Note(
            velocity=70,
            pitch=midi_note,
            start=start_time_sec,
            end=start_time_sec + duration_sec
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    return midi


def render_midi_to_audio(midi: 'pretty_midi.PrettyMIDI', sample_rate: int = 44100) -> AudioSegment:
    """
    Render a PrettyMIDI object to an AudioSegment using FluidSynth.

    Requires: fluidsynth and a soundfont installed.
    Falls back to a simple sine wave synthesis if FluidSynth unavailable.

    Instruments are distinguished by their program number:
    - Program 0 (piano): Regular keys synth for chord voicings
    - Program 38 (synth bass): Sine-like sub bass for roots
    """
    import numpy as np
    from io import BytesIO
    import wave

    try:
        # Try to use FluidSynth (requires fluidsynth and a soundfont)
        audio_data = midi.fluidsynth(fs=sample_rate)
        # Normalize
        if audio_data.max() > 0:
            audio_data = audio_data / audio_data.max() * 0.8
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
    except Exception:
        # Fallback: simple sine wave synthesis
        # Use different timbres for different instruments
        duration = midi.get_end_time()
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio_data = np.zeros(len(t), dtype=np.float32)

        for instrument in midi.instruments:
            # Check program number for timbre selection
            is_bass = instrument.program == 38  # Synth bass for roots

            for note in instrument.notes:
                freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
                note_start = int(note.start * sample_rate)
                note_end = int(note.end * sample_rate)
                if note_end > len(t):
                    note_end = len(t)
                note_t = np.arange(note_end - note_start) / sample_rate

                # Different envelope and timbre for bass vs keys
                if is_bass:
                    # Fat square-ish sub bass - punchy, present
                    envelope = np.ones(len(note_t))
                    attack_samples = min(int(0.02 * sample_rate), len(note_t))
                    release_samples = min(int(0.1 * sample_rate), len(note_t))
                    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
                    envelope[-release_samples:] *= np.linspace(1, 0, release_samples)

                    # Square wave approximation (odd harmonics: 1, 3, 5, 7)
                    # This gives that fat, present dub bass character
                    fundamental = np.sin(2 * np.pi * freq * note_t)
                    harmonic_3 = np.sin(2 * np.pi * freq * 3 * note_t) / 3
                    harmonic_5 = np.sin(2 * np.pi * freq * 5 * note_t) / 5
                    harmonic_7 = np.sin(2 * np.pi * freq * 7 * note_t) / 7

                    note_audio = 0.45 * envelope * (fundamental + harmonic_3 + harmonic_5 + harmonic_7)
                else:
                    # Piano-like synth - percussive attack, exponential decay
                    # Strong attack, then decays like a struck string
                    envelope = np.ones(len(note_t))

                    # Very fast attack (5ms) - percussive strike
                    attack_samples = min(int(0.005 * sample_rate), len(note_t))
                    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

                    # Exponential decay over the note duration
                    # Decay faster for higher notes (like a real piano)
                    decay_rate = 3.0 + (note.pitch - 60) * 0.02  # Higher notes decay faster
                    decay_rate = max(1.5, min(decay_rate, 6.0))  # Clamp between 1.5 and 6

                    decay_portion = envelope[attack_samples:]
                    decay_t = np.linspace(0, len(decay_portion) / sample_rate, len(decay_portion))
                    decay_portion[:] = np.exp(-decay_rate * decay_t)

                    # Soft release at the end
                    release_samples = min(int(0.05 * sample_rate), len(note_t))
                    envelope[-release_samples:] *= np.linspace(1, 0, release_samples)

                    # Piano-like timbre: fundamental + harmonics with decreasing amplitude
                    # Slight inharmonicity for realism (harmonics slightly sharp)
                    fundamental = np.sin(2 * np.pi * freq * note_t)
                    harmonic_2 = 0.5 * np.sin(2 * np.pi * freq * 2.001 * note_t)  # Slight detuning
                    harmonic_3 = 0.25 * np.sin(2 * np.pi * freq * 3.002 * note_t)
                    harmonic_4 = 0.125 * np.sin(2 * np.pi * freq * 4.003 * note_t)
                    harmonic_5 = 0.0625 * np.sin(2 * np.pi * freq * 5.004 * note_t)

                    note_audio = 0.15 * envelope * (
                        fundamental + harmonic_2 + harmonic_3 + harmonic_4 + harmonic_5
                    )

                audio_data[note_start:note_end] += note_audio[:note_end - note_start]

        # Normalize
        if np.abs(audio_data).max() > 0:
            audio_data = audio_data / np.abs(audio_data).max() * 0.8
        audio_int16 = (audio_data * 32767).astype(np.int16)

    # Create WAV in memory
    wav_buffer = BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    wav_buffer.seek(0)
    return AudioSegment.from_wav(wav_buffer)


def select_bassflute_events(
    chain: List[ChainLink],
    harmonic_timeline: List[HarmonicEvent],
    density: float = 0.3,  # Unused now - continuous playback
    seed: Optional[int] = None,
    total_duration_ms: float = 0
) -> List[BassFluteEvent]:
    """
    Select bass flute one-shots to layer over the chain.

    Bass flutes play CONTINUOUSLY - when one ends, the next immediately starts.
    No gaps, no overlap. Each sample is transposed LIVE as harmonic state changes.

    Args:
        chain: The sample chain
        harmonic_timeline: Timeline of harmonic events for transposition
        density: Unused (kept for API compatibility)
        seed: Random seed for reproducibility
        total_duration_ms: Total duration to fill with bass flute samples

    Returns:
        List of BassFluteEvent (transposition is calculated dynamically during render)
    """
    if seed is not None:
        random.seed(seed)

    bassflute_samples = load_bassflute_samples()
    if not bassflute_samples:
        return []

    events = []
    sample_names = list(bassflute_samples.keys())
    current_ms = 0.0  # Track where we are in the timeline
    next_sample_idx = 0

    # Continuously add bass flute events until we've filled the duration
    while current_ms < total_duration_ms:
        # Cycle through samples in order
        bf_name = sample_names[next_sample_idx % len(sample_names)]
        next_sample_idx += 1

        bf_data = bassflute_samples[bf_name]
        bf_pcs = bf_data["pitch_classes"]

        events.append(BassFluteEvent(
            sample_name=bf_name,
            start_ms=current_ms,
            original_pcs=bf_pcs,
        ))

        # Move to end of this sample (actual duration calculated during render)
        # Use original duration as estimate
        sample_duration = bf_data.get("duration_ms", 20000)
        current_ms += sample_duration

    return events


def build_harmonic_timeline(chain: List[ChainLink]) -> List[HarmonicEvent]:
    """
    Build a timeline of harmonic events from the chain.

    Each harmonic event represents when a chord is active.

    For PROGRESSION samples (glaz, hyacinthe, kraus) with chord_sequence:
      - Emits events for middle and last chords using chord_sequence timing

    For STANDARD samples (Feldman, Handel):
      - Uses chord_b_onset_ratio for timing (ignores chord_sequence)
    """
    timeline = []
    current_time_ms = 0

    for link in chain:
        # Calculate actual duration (accounting for transposition)
        if link.transposition != 0:
            rate_change = 2 ** (link.transposition / 12.0)
            actual_duration_ms = link.duration_ms / rate_change
        else:
            actual_duration_ms = link.duration_ms

        # Check if this is a PROGRESSION sample (glaz, hyacinthe, kraus)
        # These have accurate chord_sequence timing data
        # Feldman and Handel samples should use chord_b_onset_ratio instead
        is_progression_sample = (
            link.sample.startswith("glaz_") or
            link.sample.startswith("hyacinthe_") or
            link.sample.startswith("kraus_")
        )

        if is_progression_sample and link.chord_sequence and len(link.chord_sequence) > 0:
            # Emit HarmonicEvent for middle and last chords only
            # SKIP first chord (i=0): it's already a subset of the previous sample's inferred chord
            # Middle chords: infer new chord supersets
            # Last chord: use link.inferred_chord (already computed during chaining)
            prev_root = None
            prev_chord_pcs_local = None  # Local tracking for voice leading within progression
            num_chords = len(link.chord_sequence)

            for i, chord_data in enumerate(link.chord_sequence):
                # Skip first chord - it doesn't change harmonic state
                # (first_pcs is already subset of previous chord)
                # EXCEPTION: 1-chord samples still need to emit their inferred chord,
                # but at the END of the sample (not the start)
                if i == 0 and num_chords > 1:
                    continue

                # For 1-chord samples: emit at END of sample (when next sample starts)
                # For multi-chord samples: emit at the chord's actual start_time
                if num_chords == 1:
                    # 1-chord sample: event at end of sample
                    chord_start_ms = current_time_ms + actual_duration_ms
                else:
                    # Multi-chord sample: event at chord's start_time
                    chord_start_time_sec = chord_data.get("start_time", 0)

                    # Apply transposition time scaling
                    if link.transposition != 0:
                        rate_change = 2 ** (link.transposition / 12.0)
                        chord_start_time_sec = chord_start_time_sec / rate_change

                    chord_start_ms = current_time_ms + (chord_start_time_sec * 1000)

                # End time is when the NEXT chord starts (or end of sample)
                if i + 1 < num_chords:
                    next_start_sec = link.chord_sequence[i + 1].get("start_time", 0)
                    if link.transposition != 0:
                        next_start_sec = next_start_sec / rate_change
                    chord_end_ms = current_time_ms + (next_start_sec * 1000)
                else:
                    chord_end_ms = current_time_ms + actual_duration_ms

                # For LAST chord: use link.inferred_chord (the bridging chord)
                # For MIDDLE chords: infer new chord superset
                if i == num_chords - 1:
                    # Last chord - use the pre-computed bridging chord
                    chord_pcs = set(link.inferred_chord.get("pitch_classes", []))
                    chord_root = link.inferred_chord.get("root", 0)
                    chord_name = link.inferred_chord.get("name", "unknown")
                else:
                    # Middle chord - infer chord superset
                    raw_pcs = set(chord_data.get("pitch_classes", []))
                    if link.transposition != 0:
                        transposed_pcs = transpose_set(raw_pcs, link.transposition)
                    else:
                        transposed_pcs = raw_pcs

                    chord_candidates = infer_chord_supersets(
                        transposed_pcs,
                        max_results=5,
                        target_sizes=[4, 5, 6],
                        prev_root=prev_root,
                        prev_chord_pcs=prev_chord_pcs_local
                    )

                    if chord_candidates:
                        inferred = chord_candidates[0]
                        chord_pcs = set(inferred.get("pitch_classes", list(transposed_pcs)))
                        chord_root = inferred.get("root", min(transposed_pcs) if transposed_pcs else 0)
                        chord_name = inferred.get("name", f"middle_chord_{i}")
                        prev_root = chord_root  # Track for root movement
                        prev_chord_pcs_local = chord_pcs  # Track for voice leading
                    else:
                        # Fallback to raw pitch classes if no superset found
                        chord_pcs = transposed_pcs
                        chord_root = min(transposed_pcs) if transposed_pcs else 0
                        chord_name = chord_data.get("chord_name", f"middle_chord_{i}")

                timeline.append(HarmonicEvent(
                    start_ms=chord_start_ms,
                    end_ms=chord_end_ms + 5000,  # Extend for overlapping layers
                    chord_pcs=chord_pcs,
                    chord_root=chord_root,
                    chord_name=chord_name,
                ))
        else:
            # Non-progression sample (Feldman/Handel)
            # Emit TWO events: chord A at sample start, chord B at onset ratio

            # CHORD A: At sample start
            # Use the PREVIOUS sample's inferred chord (which bridges to this sample's first_pcs)
            # For the FIRST sample, infer a chord from first_pcs
            if timeline:
                # Use the last emitted chord (which bridges to this sample)
                # NOTE: Keep the ORIGINAL chord name (no suffix) so it can be found in chord_dict
                chord_a_pcs = timeline[-1].chord_pcs
                chord_a_root = timeline[-1].chord_root
                chord_a_name = timeline[-1].chord_name.replace("_cont", "")  # Strip any existing suffix
            else:
                # First sample - infer chord from first_pcs (no prev chord for voice leading)
                first_pcs_transposed = link.first_pcs
                if link.transposition != 0:
                    first_pcs_transposed = transpose_set(first_pcs_transposed, link.transposition)
                chord_a_candidates = infer_chord_supersets(
                    first_pcs_transposed, max_results=1, target_sizes=[4, 5, 6],
                    prev_root=None, prev_chord_pcs=None
                )
                if chord_a_candidates:
                    chord_a_pcs = set(chord_a_candidates[0].get("pitch_classes", list(first_pcs_transposed)))
                    chord_a_root = chord_a_candidates[0].get("root", 0)
                    chord_a_name = chord_a_candidates[0].get("name", "first_chord")
                else:
                    chord_a_pcs = first_pcs_transposed
                    chord_a_root = min(first_pcs_transposed) if first_pcs_transposed else 0
                    chord_a_name = "first_chord"

            # Check if this is a 1-chord sample
            is_one_chord = (
                link.chord_sequence and len(link.chord_sequence) == 1
            ) or (
                link.first_pcs == link.last_pcs
            )

            # Emit chord A at sample start
            chord_a_start_ms = current_time_ms
            if is_one_chord:
                chord_a_end_ms = current_time_ms + actual_duration_ms + 5000
            else:
                chord_a_end_ms = current_time_ms + (actual_duration_ms * link.chord_b_onset_ratio)

            timeline.append(HarmonicEvent(
                start_ms=chord_a_start_ms,
                end_ms=chord_a_end_ms,
                chord_pcs=chord_a_pcs,
                chord_root=chord_a_root,
                chord_name=chord_a_name,
            ))

            # CHORD B: At chord_b_onset_ratio (only for 2-chord samples)
            if not is_one_chord:
                chord_b_start_ms = current_time_ms + (actual_duration_ms * link.chord_b_onset_ratio)
                chord_b_end_ms = current_time_ms + actual_duration_ms + 5000

                chord_b_pcs = set(link.inferred_chord.get("pitch_classes", []))
                chord_b_root = link.inferred_chord.get("root", 0)
                chord_b_name = link.inferred_chord.get("name", "unknown")

                timeline.append(HarmonicEvent(
                    start_ms=chord_b_start_ms,
                    end_ms=chord_b_end_ms,
                    chord_pcs=chord_b_pcs,
                    chord_root=chord_b_root,
                    chord_name=chord_b_name,
                ))

        current_time_ms += actual_duration_ms

    return timeline


def build_harmonic_timeline_from_rendered(
    rendered_events: List[RenderedSkeletonEvent],
    chain: List[ChainLink]
) -> List[HarmonicEvent]:
    """
    Build a timeline of harmonic events from RENDERED timing.

    This is the corrected version that uses actual audio render timing
    as the single source of truth. Audio render creates the clock,
    MIDI follows that clock.

    Key difference from build_harmonic_timeline:
    - Uses rendered_start_ms, rendered_duration_ms from RenderedSkeletonEvent
    - NOT link.duration_ms from samplesData.json

    For STANDARD samples (Feldman, Handel):
      - Chord A at rendered_start_ms (continuation of previous chord)
      - Chord B at chord_b_start_ms (calculated from rendered timing)
    """
    timeline = []

    for i, re in enumerate(rendered_events):
        link = chain[i] if i < len(chain) else None

        # Check if this is a PROGRESSION sample (glaz, hyacinthe, kraus)
        is_progression_sample = (
            re.sample_name.startswith("glaz_") or
            re.sample_name.startswith("hyacinthe_") or
            re.sample_name.startswith("kraus_")
        )

        if is_progression_sample and link and link.chord_sequence and len(link.chord_sequence) > 0:
            # For progression samples, scale chord sequence timing by varispeed rate
            rate = 2 ** (re.transposition / 12.0) if re.transposition != 0 else 1.0
            num_chords = len(link.chord_sequence)
            prev_root = None
            prev_chord_pcs_local = None

            for j, chord_data in enumerate(link.chord_sequence):
                if j == 0 and num_chords > 1:
                    continue

                if num_chords == 1:
                    chord_start_ms = re.rendered_end_ms
                else:
                    source_start_sec = chord_data.get("start_time", 0)
                    # Scale by varispeed rate
                    rendered_offset_ms = (source_start_sec * 1000) / rate
                    chord_start_ms = re.rendered_start_ms + rendered_offset_ms

                # End time
                if j + 1 < num_chords:
                    next_start_sec = link.chord_sequence[j + 1].get("start_time", 0)
                    next_rendered_offset_ms = (next_start_sec * 1000) / rate
                    chord_end_ms = re.rendered_start_ms + next_rendered_offset_ms
                else:
                    chord_end_ms = re.rendered_end_ms + 5000

                # Get chord info
                if j == num_chords - 1:
                    chord_pcs = set(link.inferred_chord.get("pitch_classes", []))
                    chord_root = link.inferred_chord.get("root", 0)
                    chord_name = link.inferred_chord.get("name", "unknown")
                else:
                    raw_pcs = set(chord_data.get("pitch_classes", []))
                    if re.transposition != 0:
                        transposed_pcs = transpose_set(raw_pcs, re.transposition)
                    else:
                        transposed_pcs = raw_pcs

                    chord_candidates = infer_chord_supersets(
                        transposed_pcs, max_results=5, target_sizes=[4, 5, 6],
                        prev_root=prev_root, prev_chord_pcs=prev_chord_pcs_local
                    )
                    if chord_candidates:
                        inferred = chord_candidates[0]
                        chord_pcs = set(inferred.get("pitch_classes", list(transposed_pcs)))
                        chord_root = inferred.get("root", min(transposed_pcs) if transposed_pcs else 0)
                        chord_name = inferred.get("name", f"middle_chord_{j}")
                        prev_root = chord_root
                        prev_chord_pcs_local = chord_pcs
                    else:
                        chord_pcs = transposed_pcs
                        chord_root = min(transposed_pcs) if transposed_pcs else 0
                        chord_name = chord_data.get("chord_name", f"middle_chord_{j}")

                timeline.append(HarmonicEvent(
                    start_ms=chord_start_ms,
                    end_ms=chord_end_ms,
                    chord_pcs=chord_pcs,
                    chord_root=chord_root,
                    chord_name=chord_name,
                ))
        else:
            # Non-progression sample (Feldman/Handel)
            # Emit TWO events: chord A at sample start, chord B at onset ratio

            # CHORD A: At rendered sample start
            # Use the previous chord's info (continuation of that chord through sample start)
            # NOTE: Keep the ORIGINAL chord name (no suffix) so it can be found in chord_dict
            if timeline:
                chord_a_pcs = timeline[-1].chord_pcs
                chord_a_root = timeline[-1].chord_root
                chord_a_name = timeline[-1].chord_name.replace("_cont", "")  # Strip any existing suffix
            else:
                # First sample - infer chord from first_pcs (no prev chord for voice leading)
                first_pcs_transposed = re.first_pcs
                chord_a_candidates = infer_chord_supersets(
                    first_pcs_transposed, max_results=1, target_sizes=[4, 5, 6],
                    prev_root=None, prev_chord_pcs=None
                )
                if chord_a_candidates:
                    chord_a_pcs = set(chord_a_candidates[0].get("pitch_classes", list(first_pcs_transposed)))
                    chord_a_root = chord_a_candidates[0].get("root", 0)
                    chord_a_name = chord_a_candidates[0].get("name", "first_chord")
                else:
                    chord_a_pcs = first_pcs_transposed
                    chord_a_root = min(first_pcs_transposed) if first_pcs_transposed else 0
                    chord_a_name = "first_chord"

            # Check if this is a 1-chord sample
            is_one_chord = re.first_pcs == re.last_pcs

            # Emit chord A using RENDERED timing
            chord_a_start_ms = re.chord_a_start_ms
            if is_one_chord:
                chord_a_end_ms = re.rendered_end_ms + 5000
            else:
                chord_a_end_ms = re.chord_b_start_ms

            timeline.append(HarmonicEvent(
                start_ms=chord_a_start_ms,
                end_ms=chord_a_end_ms,
                chord_pcs=chord_a_pcs,
                chord_root=chord_a_root,
                chord_name=chord_a_name,
            ))

            # CHORD B: At chord_b_start_ms (calculated from rendered timing)
            if not is_one_chord:
                chord_b_pcs = set(re.inferred_chord.get("pitch_classes", []))
                chord_b_root = re.inferred_chord.get("root", 0)
                chord_b_name = re.inferred_chord.get("name", "unknown")

                timeline.append(HarmonicEvent(
                    start_ms=re.chord_b_start_ms,
                    end_ms=re.rendered_end_ms + 5000,
                    chord_pcs=chord_b_pcs,
                    chord_root=chord_b_root,
                    chord_name=chord_b_name,
                ))

    return timeline


@dataclass
class BroderoEvent:
    """A Brodero one-shot with timing info."""
    sample_name: str
    start_ms: float
    original_pcs: Set[int]
    audio_path: Path


def render_brodero_layer(
    harmonic_timeline: List[HarmonicEvent],
    total_duration_ms: float,
    min_silence_seconds: float = 10.0,
    max_silence_seconds: float = 15.0,
    cloud_duration_seconds: float = 4.0,
    bpm: float = 120.0,
    verbose: bool = True,
    seed: Optional[int] = None
) -> AudioSegment:
    """
    Render Brodero samples as sporadic clouds of 16th notes.

    Clouds emerge after random stretches of silence (10-15 seconds).
    Each cloud lasts about 4 seconds of continuous 16th notes.
    Each sample uses REACTIVE transposition to fit the current chord.
    """
    if seed is not None:
        random.seed(seed + 888)  # Different seed offset for variety

    brodero_samples = load_brodero_samples()
    if not brodero_samples:
        return AudioSegment.silent(duration=int(total_duration_ms))

    layer = AudioSegment.silent(duration=int(total_duration_ms))

    # 16th note interval at given BPM
    beat_ms = 60000.0 / bpm  # Quarter note
    sixteenth_ms = beat_ms / 4  # 16th note (125ms at 120 BPM)

    # Calculate notes per cloud based on cloud duration
    cloud_duration_ms = cloud_duration_seconds * 1000.0
    notes_per_cloud = int(cloud_duration_ms / sixteenth_ms)  # ~32 notes for 4 seconds

    if verbose:
        print(f"    Brodero: {len(brodero_samples)} samples")
        print(f"    Clouds: ~{notes_per_cloud} 16th notes ({cloud_duration_seconds}s) after {min_silence_seconds}-{max_silence_seconds}s silence")
        print(f"    (16th note = {sixteenth_ms:.0f}ms at {bpm} BPM)")

    gliss_count = 0
    total_notes = 0
    num_clouds = 0
    next_sample_idx = 0

    # Start with a random silence period
    current_ms = random.uniform(min_silence_seconds, max_silence_seconds) * 1000.0

    while current_ms < total_duration_ms:
        # This is the start of a cloud
        cloud_start_ms = current_ms
        num_clouds += 1

        # Render notes_per_cloud 16th notes in this cloud
        for note_idx in range(notes_per_cloud):
            note_start_ms = cloud_start_ms + (note_idx * sixteenth_ms)

            if note_start_ms >= total_duration_ms:
                break

            # Cycle through samples
            sample_data = brodero_samples[next_sample_idx % len(brodero_samples)]
            next_sample_idx += 1

            sample_pcs = sample_data["pitch_classes"]
            audio_path = sample_data["audio_path"]

            try:
                audio, gliss_log = render_sample_with_live_transposition(
                    audio_path=audio_path,
                    sample_pcs=sample_pcs,
                    start_ms=note_start_ms,
                    harmonic_timeline=harmonic_timeline,
                    verbose=False,
                    sample_name=sample_data["name"]
                )

                if len(audio) == 0:
                    continue

                gliss_count += len(gliss_log)
                total_notes += 1

                # Overlay at position
                position_ms = int(note_start_ms)
                if position_ms + len(audio) > len(layer):
                    extra = (position_ms + len(audio)) - len(layer)
                    layer = layer + AudioSegment.silent(duration=int(extra))

                layer = layer.overlay(audio, position=position_ms)

            except Exception as e:
                if verbose:
                    print(f"      Error loading {sample_data['name']}: {e}")

        # Move past the cloud, then add random silence before next cloud
        current_ms = cloud_start_ms + cloud_duration_ms
        current_ms += random.uniform(min_silence_seconds, max_silence_seconds) * 1000.0

    if verbose:
        print(f"    Brodero: {num_clouds} clouds, {total_notes} total notes, {gliss_count} live glissandos")

    return layer


def render_organetta_layer(
    harmonic_timeline: List[HarmonicEvent],
    total_duration_ms: float,
    interval_seconds: float = 4.0,
    verbose: bool = True
) -> AudioSegment:
    """
    Render Organetta samples as a layer.

    Fires samples in alphanumeric order, looping through entire library,
    at the specified interval (default: every 4 seconds).
    Each sample uses REACTIVE transposition.

    If the scheduled sample can't fit the current harmonic state, skips to
    the next sample that does fit. If none fit, drops out until the next slot.
    """
    organetta_samples = load_organetta_samples()
    if not organetta_samples:
        return AudioSegment.silent(duration=int(total_duration_ms))

    layer = AudioSegment.silent(duration=int(total_duration_ms))

    interval_ms = interval_seconds * 1000.0
    num_fires = int(total_duration_ms / interval_ms)

    if verbose:
        print(f"    Organetta: {len(organetta_samples)} samples, every {interval_seconds}s, {num_fires} total slots")
        print(f"    (Reactive transposition + skip-to-fitting-sample enabled)")

    gliss_count = 0
    played_count = 0
    skipped_count = 0
    next_sample_idx = 0  # Track position in sample sequence

    for i in range(num_fires):
        start_ms = i * interval_ms

        # Get current harmonic state
        he = get_harmonic_event_at_time(start_ms, harmonic_timeline)
        if he is None:
            skipped_count += 1
            continue

        # Find a sample that fits, starting from current position
        result = find_next_fitting_oneshot(organetta_samples, next_sample_idx, he.chord_pcs)
        if result is None:
            # No sample fits current chord - drop out
            skipped_count += 1
            continue

        sample_idx, sample_data = result
        next_sample_idx = (sample_idx + 1) % len(organetta_samples)  # Advance for next time

        sample_pcs = sample_data["pitch_classes"]
        audio_path = sample_data["audio_path"]

        try:
            audio, gliss_log = render_sample_with_live_transposition(
                audio_path=audio_path,
                sample_pcs=sample_pcs,
                start_ms=start_ms,
                harmonic_timeline=harmonic_timeline,
                verbose=False,
                sample_name=sample_data["name"]
            )

            if len(audio) == 0:
                continue

            gliss_count += len(gliss_log)
            played_count += 1

            position_ms = int(start_ms)
            if position_ms + len(audio) > len(layer):
                extra = (position_ms + len(audio)) - len(layer)
                layer = layer + AudioSegment.silent(duration=int(extra))

            layer = layer.overlay(audio, position=position_ms)

        except Exception as e:
            if verbose:
                print(f"      Error loading {sample_data['name']}: {e}")

    if verbose:
        print(f"    Organetta: {played_count} played, {skipped_count} dropped out, {gliss_count} glissandos")

    return layer


def render_minorchordbeat_layer(
    harmonic_timeline: List[HarmonicEvent],
    total_duration_ms: float,
    bpm: float = 120.0,
    verbose: bool = True
) -> AudioSegment:
    """
    Render MinorChordBeat samples as a rhythmic layer.

    Fires samples in alphanumeric order, looping through library,
    at eighth note intervals (based on BPM).
    Each sample uses REACTIVE transposition.

    If the scheduled sample can't fit, skips to next fitting sample.
    If none fit, drops out until the next slot.
    """
    samples = load_minorchordbeat_samples()
    if not samples:
        return AudioSegment.silent(duration=int(total_duration_ms))

    layer = AudioSegment.silent(duration=int(total_duration_ms))

    # Eighth note interval at given BPM
    beat_ms = 60000.0 / bpm  # Quarter note
    interval_ms = beat_ms / 2  # Eighth note
    num_fires = int(total_duration_ms / interval_ms)

    if verbose:
        print(f"    MinorChordBeat: {len(samples)} samples, eighth notes at {bpm} BPM ({interval_ms:.0f}ms), {num_fires} slots")
        print(f"    (Reactive transposition + skip-to-fitting-sample enabled)")

    gliss_count = 0
    played_count = 0
    skipped_count = 0
    next_sample_idx = 0

    for i in range(num_fires):
        start_ms = i * interval_ms

        # Always play next sample in sequence (no drop-out)
        sample_data = samples[next_sample_idx]
        next_sample_idx = (next_sample_idx + 1) % len(samples)

        sample_pcs = sample_data["pitch_classes"]
        audio_path = sample_data["audio_path"]

        try:
            audio, gliss_log = render_sample_with_live_transposition(
                audio_path=audio_path,
                sample_pcs=sample_pcs,
                start_ms=start_ms,
                harmonic_timeline=harmonic_timeline,
                verbose=False,
                sample_name=sample_data["name"]
            )

            if len(audio) == 0:
                continue

            gliss_count += len(gliss_log)
            played_count += 1

            position_ms = int(start_ms)
            if position_ms + len(audio) > len(layer):
                extra = (position_ms + len(audio)) - len(layer)
                layer = layer + AudioSegment.silent(duration=int(extra))

            layer = layer.overlay(audio, position=position_ms)

        except Exception as e:
            if verbose:
                print(f"      Error loading {sample_data['name']}: {e}")

    if verbose:
        print(f"    MinorChordBeat: {played_count} played, {gliss_count} glissandos")

    return layer


def render_mutebowl_layer(
    harmonic_timeline: List[HarmonicEvent],
    total_duration_ms: float,
    bpm: float = 120.0,
    verbose: bool = True
) -> AudioSegment:
    """
    Render MuteBowl samples as a rhythmic layer.

    Fires samples in alphanumeric order, looping through library,
    at eighth note intervals (based on BPM).
    Each sample uses REACTIVE transposition.

    If the scheduled sample can't fit, skips to next fitting sample.
    If none fit, drops out until the next slot.
    """
    samples = load_mutebowl_samples()
    if not samples:
        return AudioSegment.silent(duration=int(total_duration_ms))

    layer = AudioSegment.silent(duration=int(total_duration_ms))

    # Eighth note interval at given BPM
    beat_ms = 60000.0 / bpm  # Quarter note
    interval_ms = beat_ms / 2  # Eighth note
    num_fires = int(total_duration_ms / interval_ms)

    if verbose:
        print(f"    MuteBowl: {len(samples)} samples, eighth notes at {bpm} BPM ({interval_ms:.0f}ms), {num_fires} slots")
        print(f"    (Reactive transposition + skip-to-fitting-sample enabled)")

    gliss_count = 0
    played_count = 0
    next_sample_idx = 0

    for i in range(num_fires):
        start_ms = i * interval_ms

        # Always play next sample in sequence (no drop-out)
        sample_data = samples[next_sample_idx]
        next_sample_idx = (next_sample_idx + 1) % len(samples)

        sample_pcs = sample_data["pitch_classes"]
        audio_path = sample_data["audio_path"]

        try:
            audio, gliss_log = render_sample_with_live_transposition(
                audio_path=audio_path,
                sample_pcs=sample_pcs,
                start_ms=start_ms,
                harmonic_timeline=harmonic_timeline,
                verbose=False,
                sample_name=sample_data["name"]
            )

            if len(audio) == 0:
                continue

            gliss_count += len(gliss_log)
            played_count += 1

            position_ms = int(start_ms)
            if position_ms + len(audio) > len(layer):
                extra = (position_ms + len(audio)) - len(layer)
                layer = layer + AudioSegment.silent(duration=int(extra))

            layer = layer.overlay(audio, position=position_ms)

        except Exception as e:
            if verbose:
                print(f"      Error loading {sample_data['name']}: {e}")

    if verbose:
        print(f"    MuteBowl: {played_count} played, {gliss_count} glissandos")

    return layer


def render_qualitychords_layer(
    harmonic_timeline: List[HarmonicEvent],
    total_duration_ms: float,
    bpm: float = 120.0,
    sparsity: float = 0.4,  # Probability of playing on each beat (40% = sporadic)
    seed: Optional[int] = None,
    verbose: bool = True
) -> AudioSegment:
    """
    Render quality-aware chord samples as a sporadic rhythmic layer.

    Uses minor-chords samples when current chord is minor,
    major-chords samples when current chord is major.
    Plays sporadically (not every beat) for texture.
    Each sample uses REACTIVE transposition.
    """
    minor_samples = load_minorchords_samples()
    major_samples = load_majorchords_samples()

    if not minor_samples and not major_samples:
        return AudioSegment.silent(duration=int(total_duration_ms))

    layer = AudioSegment.silent(duration=int(total_duration_ms))

    # Eighth note interval at given BPM
    beat_ms = 60000.0 / bpm  # Quarter note
    interval_ms = beat_ms / 2  # Eighth note
    num_slots = int(total_duration_ms / interval_ms)

    # Use seed for reproducible sporadic pattern
    if seed is not None:
        random.seed(seed + 999)  # Offset to get different pattern than other layers

    if verbose:
        print(f"    QualityChords: {len(minor_samples)} minor, {len(major_samples)} major samples")
        print(f"    Eighth notes at {bpm} BPM, ~{sparsity:.0%} density (sporadic)")

    gliss_count = 0
    played_count = 0
    minor_count = 0
    major_count = 0
    next_minor_idx = 0
    next_major_idx = 0

    for i in range(num_slots):
        # Sporadic: randomly skip some beats
        if random.random() > sparsity:
            continue

        start_ms = i * interval_ms

        # Get current harmonic event to determine chord quality
        he = get_harmonic_event_at_time(start_ms, harmonic_timeline)
        if he is None:
            continue

        # Look up chord quality
        chord_name = he.chord_name if hasattr(he, 'chord_name') else ""
        quality = get_chord_quality(chord_name)

        # Select sample based on quality
        if quality == "minor" and minor_samples:
            sample_data = minor_samples[next_minor_idx]
            next_minor_idx = (next_minor_idx + 1) % len(minor_samples)
            minor_count += 1
        elif quality == "major" and major_samples:
            sample_data = major_samples[next_major_idx]
            next_major_idx = (next_major_idx + 1) % len(major_samples)
            major_count += 1
        elif minor_samples:
            # Default to minor for "other" qualities (dominant, dim, etc)
            sample_data = minor_samples[next_minor_idx]
            next_minor_idx = (next_minor_idx + 1) % len(minor_samples)
            minor_count += 1
        else:
            continue

        sample_pcs = sample_data["pitch_classes"]
        audio_path = sample_data["audio_path"]

        try:
            audio, gliss_log = render_sample_with_live_transposition(
                audio_path=audio_path,
                sample_pcs=sample_pcs,
                start_ms=start_ms,
                harmonic_timeline=harmonic_timeline,
                verbose=False,
                sample_name=sample_data["name"]
            )

            if len(audio) == 0:
                continue

            # Normalize to prevent clipping
            audio = normalize_audio_peak(audio, -6.0)

            gliss_count += len(gliss_log)
            played_count += 1

            position_ms = int(start_ms)
            if position_ms + len(audio) > len(layer):
                extra = (position_ms + len(audio)) - len(layer)
                layer = layer + AudioSegment.silent(duration=int(extra))

            layer = layer.overlay(audio, position=position_ms)

        except Exception as e:
            if verbose:
                print(f"      Error loading {sample_data['name']}: {e}")

    if verbose:
        print(f"    QualityChords: {played_count} played ({minor_count} minor, {major_count} major), {gliss_count} glissandos")

    return layer


# NOTE: render_progression_layer has been removed.
# Progressions (glaz_sax, hyacinthe, kraus) are skeleton samples that participate
# in chain building and are concatenated sequentially. They are not overlay layers.
# Their chord_sequence events are added to the harmonic timeline during chain render.


def load_jicello_samples() -> List[Dict]:
    """
    Load Jicello expanded samples metadata.
    Returns a list for distributed playback.
    """
    global JICELLO_SAMPLES
    if JICELLO_SAMPLES:
        return JICELLO_SAMPLES

    if not JICELLO_MANIFEST_PATH.exists():
        print(f"Warning: Jicello manifest not found at {JICELLO_MANIFEST_PATH}")
        return []

    with open(JICELLO_MANIFEST_PATH) as f:
        raw_data = json.load(f)

    # Note name to pitch class mapping
    note_to_pc = {
        'c': 0, 'cs': 1, 'df': 1, 'd': 2, 'ds': 3, 'ef': 3, 'e': 4,
        'f': 5, 'fs': 6, 'gf': 6, 'g': 7, 'gs': 8, 'af': 8, 'a': 9,
        'as': 10, 'bf': 10, 'b': 11
    }

    for sample_name, sample_data in raw_data.items():
        if sample_name.startswith("_"):
            continue

        note_names = sample_data.get("note_names", [])
        pitch_classes = []
        for note in note_names:
            note_lower = note.lower()
            if note_lower in note_to_pc:
                pitch_classes.append(note_to_pc[note_lower])

        if pitch_classes:
            audio_path = JICELLO_AUDIO_DIR / f"{sample_name}.wav"
            if audio_path.exists():
                JICELLO_SAMPLES.append({
                    "name": sample_name,
                    "pitch_classes": set(pitch_classes),
                    "note_names": note_names,
                    "audio_path": audio_path,
                })

    # Sort by name
    JICELLO_SAMPLES.sort(key=lambda x: x["name"])
    return JICELLO_SAMPLES


def render_sample_with_live_transposition(
    audio_path: Path,
    sample_pcs: Set[int],
    start_ms: float,
    harmonic_timeline: List[HarmonicEvent],
    verbose: bool = False,
    sample_name: str = ""
) -> Tuple[AudioSegment, List[str]]:
    """
    Render any sample with REACTIVE transposition as harmonic state changes.

    Simple reactive approach: at each moment, check the harmonic state and
    transpose if needed. If the sample can't fit the current harmony, it fades out.

    Returns (processed_audio, gliss_log_messages)
    """
    if not audio_path.exists():
        return AudioSegment.empty(), []

    original_audio = AudioSegment.from_file(audio_path)

    # Use the generic reactive transposition function
    result, output_duration, gliss_count = render_overlay_with_reactive_transposition(
        audio=original_audio,
        original_pcs=sample_pcs,
        start_ms=start_ms,
        harmonic_timeline=harmonic_timeline,
        find_transposition_fn=find_bassflute_transposition,
        verbose=verbose,
        sample_name=sample_name
    )

    # Return gliss count as list for compatibility
    return result, [f"glissando" for _ in range(gliss_count)]


def render_jicello_layer(
    harmonic_timeline: List[HarmonicEvent],
    total_duration_ms: float,
    verbose: bool = True
) -> AudioSegment:
    """
    Render Jicello samples CONTINUOUSLY (like bass flute).

    When one sample ends, the next immediately starts. No gaps, no overlap.
    Each sample uses LIVE transposition with anticipatory glissando -
    re-transposes when harmonic context changes during playback.
    """
    jicello_samples = load_jicello_samples()
    if not jicello_samples:
        return AudioSegment.silent(duration=int(total_duration_ms))

    # Create silent base
    layer = AudioSegment.silent(duration=int(total_duration_ms))

    num_samples = len(jicello_samples)

    if verbose:
        print(f"    Jicello: {num_samples} samples, CONTINUOUS playback (no gaps)")
        print(f"    (Reactive transposition enabled)")

    gliss_count = 0
    total_played = 0
    current_ms = 0.0
    sample_idx = 0

    while current_ms < total_duration_ms:
        sample_data = jicello_samples[sample_idx % num_samples]
        sample_idx += 1
        start_ms = current_ms
        sample_pcs = sample_data["pitch_classes"]
        audio_path = sample_data["audio_path"]

        try:
            audio, gliss_log = render_sample_with_live_transposition(
                audio_path=audio_path,
                sample_pcs=sample_pcs,
                start_ms=start_ms,
                harmonic_timeline=harmonic_timeline,
                verbose=False,
                sample_name=sample_data["name"]
            )

            if len(audio) == 0:
                # Skip to next sample if this one can't play
                continue

            gliss_count += len(gliss_log)
            total_played += 1

            # Log
            if verbose:
                # Find initial transposition
                init_trans = 0
                for he in harmonic_timeline:
                    if he.start_ms <= start_ms < he.end_ms:
                        t = find_bassflute_transposition(sample_pcs, he.chord_pcs, he.chord_root)
                        if t is not None:
                            init_trans = t
                        break
                trans_str = f" trans {init_trans:+d}" if init_trans != 0 else ""
                print(f"      {sample_data['name']} at {start_ms/1000:.2f}s{trans_str}")
                for log_line in gliss_log:
                    print(log_line)

            # Overlay at position
            position_ms = int(start_ms)
            if position_ms + len(audio) > len(layer):
                extra = (position_ms + len(audio)) - len(layer)
                layer = layer + AudioSegment.silent(duration=int(extra))

            layer = layer.overlay(audio, position=position_ms)

            # Move to end of this sample for next one (continuous playback)
            current_ms = start_ms + len(audio)

        except Exception as e:
            if verbose:
                print(f"      Error loading {sample_data['name']}: {e}")

    if verbose:
        print(f"    Jicello: {total_played} samples played continuously, {gliss_count} glissandos")

    return layer


def render_prophetfalse_layer(
    harmonic_timeline: List[HarmonicEvent],
    total_duration_ms: float,
    interval_seconds: float = 3.0,
    verbose: bool = True
) -> AudioSegment:
    """
    Render Prophet False samples as a layer.

    Fires single-note synth samples in sequence at the specified interval.
    Each sample uses REACTIVE transposition to fit the current chord.
    """
    samples = load_prophetfalse_samples()
    if not samples:
        return AudioSegment.silent(duration=int(total_duration_ms))

    layer = AudioSegment.silent(duration=int(total_duration_ms))

    interval_ms = interval_seconds * 1000.0
    num_fires = int(total_duration_ms / interval_ms)

    if verbose:
        print(f"    ProphetFalse: {len(samples)} samples, every {interval_seconds}s, {num_fires} total slots")
        print(f"    (Reactive transposition + skip-to-fitting-sample enabled)")

    gliss_count = 0
    played_count = 0
    skipped_count = 0
    next_sample_idx = 0

    for i in range(num_fires):
        start_ms = i * interval_ms

        # Get current harmonic state
        he = get_harmonic_event_at_time(start_ms, harmonic_timeline)
        if he is None:
            skipped_count += 1
            continue

        # Find a sample that fits, starting from current position
        result = find_next_fitting_oneshot(samples, next_sample_idx, he.chord_pcs)
        if result is None:
            # No sample fits current chord - drop out
            skipped_count += 1
            continue

        sample_idx, sample_data = result
        next_sample_idx = (sample_idx + 1) % len(samples)

        sample_pcs = sample_data["pitch_classes"]
        audio_path = sample_data["audio_path"]

        try:
            audio, gliss_log = render_sample_with_live_transposition(
                audio_path=audio_path,
                sample_pcs=sample_pcs,
                start_ms=start_ms,
                harmonic_timeline=harmonic_timeline,
                verbose=False,
                sample_name=sample_data["name"]
            )

            if len(audio) == 0:
                continue

            gliss_count += len(gliss_log)
            played_count += 1

            position_ms = int(start_ms)
            if position_ms + len(audio) > len(layer):
                extra = (position_ms + len(audio)) - len(layer)
                layer = layer + AudioSegment.silent(duration=int(extra))

            layer = layer.overlay(audio, position=position_ms)

        except Exception as e:
            if verbose:
                print(f"      Error loading {sample_data['name']}: {e}")

    if verbose:
        print(f"    ProphetFalse: {played_count} played, {skipped_count} dropped out, {gliss_count} glissandos")

    return layer


def render_harmonicker_layer(
    harmonic_timeline: List[HarmonicEvent],
    total_duration_ms: float,
    interval_seconds: float = 5.0,  # Unused now - continuous playback
    verbose: bool = True
) -> AudioSegment:
    """
    Render Harmonicker samples as a CONTINUOUS layer.

    Samples play back-to-back with no gaps and no overlap.
    When one sample ends, the next immediately starts.
    Each sample uses REACTIVE transposition to fit the current chord.
    """
    samples = load_harmonicker_samples()
    if not samples:
        return AudioSegment.silent(duration=int(total_duration_ms))

    layer = AudioSegment.silent(duration=int(total_duration_ms))

    if verbose:
        print(f"    Harmonicker: {len(samples)} samples, CONTINUOUS playback (no gaps)")
        print(f"    (Reactive transposition enabled)")

    gliss_count = 0
    played_count = 0
    current_ms = 0.0  # Track where we are in the timeline
    next_sample_idx = 0

    while current_ms < total_duration_ms:
        # Get current sample in sequence
        sample_data = samples[next_sample_idx]
        next_sample_idx = (next_sample_idx + 1) % len(samples)

        sample_pcs = sample_data["pitch_classes"]
        audio_path = sample_data["audio_path"]

        try:
            audio, gliss_log = render_sample_with_live_transposition(
                audio_path=audio_path,
                sample_pcs=sample_pcs,
                start_ms=current_ms,
                harmonic_timeline=harmonic_timeline,
                verbose=False,
                sample_name=sample_data["name"]
            )

            if len(audio) == 0:
                # If sample couldn't render, skip ahead a bit to avoid infinite loop
                current_ms += 1000
                continue

            gliss_count += len(gliss_log)
            played_count += 1

            position_ms = int(current_ms)
            if position_ms + len(audio) > len(layer):
                extra = (position_ms + len(audio)) - len(layer)
                layer = layer + AudioSegment.silent(duration=int(extra))

            layer = layer.overlay(audio, position=position_ms)

            # Move to end of this sample (no gap)
            current_ms += len(audio)

        except Exception as e:
            if verbose:
                print(f"      Error loading {sample_data['name']}: {e}")
            current_ms += 1000  # Skip ahead on error

    if verbose:
        print(f"    Harmonicker: {played_count} samples played continuously, {gliss_count} glissandos")

    return layer


def render_gothicharp_layer(
    harmonic_timeline: List[HarmonicEvent],
    total_duration_ms: float,
    min_silence_seconds: float = 7.0,
    max_silence_seconds: float = 9.0,
    cloud_duration_seconds: float = 5.0,
    bpm: float = 120.0,
    verbose: bool = True,
    seed: Optional[int] = None
) -> AudioSegment:
    """
    Render Gothic Harp samples as sporadic clouds of 16th notes.

    Clouds emerge after random stretches of silence (7-9 seconds).
    Each cloud lasts about 5 seconds of continuous 16th notes.
    Each note uses REACTIVE transposition to fit the current chord.
    """
    if seed is not None:
        random.seed(seed + 999)  # Different seed offset for variety

    samples = load_gothicharp_samples()
    if not samples:
        return AudioSegment.silent(duration=int(total_duration_ms))

    layer = AudioSegment.silent(duration=int(total_duration_ms))

    # 16th note interval at given BPM
    beat_ms = 60000.0 / bpm  # Quarter note
    sixteenth_ms = beat_ms / 4  # 16th note (125ms at 120 BPM)

    # Calculate notes per cloud based on cloud duration
    cloud_duration_ms = cloud_duration_seconds * 1000.0
    notes_per_cloud = int(cloud_duration_ms / sixteenth_ms)  # ~40 notes for 5 seconds

    if verbose:
        print(f"    GothicHarp: {len(samples)} samples")
        print(f"    Clouds: ~{notes_per_cloud} 16th notes ({cloud_duration_seconds}s) after {min_silence_seconds}-{max_silence_seconds}s silence")
        print(f"    (16th note = {sixteenth_ms:.0f}ms at {bpm} BPM)")

    gliss_count = 0
    total_notes = 0
    num_clouds = 0
    next_sample_idx = 0

    # Start with a random silence period
    current_ms = random.uniform(min_silence_seconds, max_silence_seconds) * 1000.0

    while current_ms < total_duration_ms:
        # This is the start of a cloud
        cloud_start_ms = current_ms
        num_clouds += 1

        # Render notes_per_cloud 16th notes in this cloud
        for note_idx in range(notes_per_cloud):
            note_start_ms = cloud_start_ms + (note_idx * sixteenth_ms)

            if note_start_ms >= total_duration_ms:
                break

            # Cycle through samples
            sample_data = samples[next_sample_idx % len(samples)]
            next_sample_idx += 1

            sample_pcs = sample_data["pitch_classes"]
            audio_path = sample_data["audio_path"]

            try:
                audio, gliss_log = render_sample_with_live_transposition(
                    audio_path=audio_path,
                    sample_pcs=sample_pcs,
                    start_ms=note_start_ms,
                    harmonic_timeline=harmonic_timeline,
                    verbose=False,
                    sample_name=sample_data["name"]
                )

                if len(audio) == 0:
                    continue

                gliss_count += len(gliss_log)
                total_notes += 1

                position_ms = int(note_start_ms)
                if position_ms + len(audio) > len(layer):
                    extra = (position_ms + len(audio)) - len(layer)
                    layer = layer + AudioSegment.silent(duration=int(extra))

                layer = layer.overlay(audio, position=position_ms)

            except Exception as e:
                if verbose:
                    print(f"      Error loading {sample_data['name']}: {e}")

        # Move past the cloud, then add random silence before next cloud
        current_ms = cloud_start_ms + cloud_duration_ms
        silence_duration = random.uniform(min_silence_seconds, max_silence_seconds) * 1000.0
        current_ms += silence_duration

    if verbose:
        print(f"    GothicHarp: {total_notes} notes played in {num_clouds} clouds, {gliss_count} glissandos")

    return layer


def render_gentleharpsi_layer(
    harmonic_timeline: List[HarmonicEvent],
    total_duration_ms: float,
    min_silence_seconds: float = 10.0,
    max_silence_seconds: float = 11.0,
    cloud_duration_seconds: float = 6.0,
    bpm: float = 120.0,
    verbose: bool = True,
    seed: Optional[int] = None
) -> AudioSegment:
    """
    Render Gentle Harpsichord samples as sporadic clouds of 32nd notes.

    Clouds emerge after random stretches of silence (10-11 seconds).
    Each cloud lasts about 6 seconds of continuous 32nd notes.
    Each note uses REACTIVE transposition to fit the current chord.
    """
    if seed is not None:
        random.seed(seed + 888)  # Different seed offset for variety

    samples = load_gentleharpsi_samples()
    if not samples:
        return AudioSegment.silent(duration=int(total_duration_ms))

    layer = AudioSegment.silent(duration=int(total_duration_ms))

    # 32nd note interval at given BPM
    beat_ms = 60000.0 / bpm  # Quarter note
    thirtysecond_ms = beat_ms / 8  # 32nd note (62.5ms at 120 BPM)

    # Calculate notes per cloud based on cloud duration
    cloud_duration_ms = cloud_duration_seconds * 1000.0
    notes_per_cloud = int(cloud_duration_ms / thirtysecond_ms)  # ~96 notes for 6 seconds

    if verbose:
        print(f"    GentleHarpsi: {len(samples)} samples")
        print(f"    Clouds: ~{notes_per_cloud} 32nd notes ({cloud_duration_seconds}s) after {min_silence_seconds}-{max_silence_seconds}s silence")
        print(f"    (32nd note = {thirtysecond_ms:.0f}ms at {bpm} BPM)")

    gliss_count = 0
    total_notes = 0
    num_clouds = 0
    next_sample_idx = 0

    # Start with a random silence period
    current_ms = random.uniform(min_silence_seconds, max_silence_seconds) * 1000.0

    while current_ms < total_duration_ms:
        # This is the start of a cloud
        cloud_start_ms = current_ms
        num_clouds += 1

        # Render notes_per_cloud 32nd notes in this cloud
        for note_idx in range(notes_per_cloud):
            note_start_ms = cloud_start_ms + (note_idx * thirtysecond_ms)

            if note_start_ms >= total_duration_ms:
                break

            # Cycle through samples
            sample_data = samples[next_sample_idx % len(samples)]
            next_sample_idx += 1

            sample_pcs = sample_data["pitch_classes"]
            audio_path = sample_data["audio_path"]

            try:
                audio, gliss_log = render_sample_with_live_transposition(
                    audio_path=audio_path,
                    sample_pcs=sample_pcs,
                    start_ms=note_start_ms,
                    harmonic_timeline=harmonic_timeline,
                    verbose=False,
                    sample_name=sample_data["name"]
                )

                if len(audio) == 0:
                    continue

                gliss_count += len(gliss_log)
                total_notes += 1

                position_ms = int(note_start_ms)
                if position_ms + len(audio) > len(layer):
                    extra = (position_ms + len(audio)) - len(layer)
                    layer = layer + AudioSegment.silent(duration=int(extra))

                layer = layer.overlay(audio, position=position_ms)

            except Exception as e:
                if verbose:
                    print(f"      Error loading {sample_data['name']}: {e}")

        # Move past the cloud, then add random silence before next cloud
        current_ms = cloud_start_ms + cloud_duration_ms
        silence_duration = random.uniform(min_silence_seconds, max_silence_seconds) * 1000.0
        current_ms += silence_duration

    if verbose:
        print(f"    GentleHarpsi: {total_notes} notes played in {num_clouds} clouds, {gliss_count} glissandos")

    return layer


def render_feedback_layer(
    harmonic_timeline: List[HarmonicEvent],
    total_duration_ms: float,
    interval_seconds: float = 8.0,
    verbose: bool = True,
    seed: Optional[int] = None
) -> AudioSegment:
    """
    Render Feedback samples as overlapping loops with random selection.

    Samples can overlap and are selected randomly.
    Each sample uses REACTIVE transposition to fit the current chord.
    """
    if seed is not None:
        random.seed(seed + 777)  # Different seed offset for variety

    samples = load_feedback_samples()
    if not samples:
        return AudioSegment.silent(duration=int(total_duration_ms))

    layer = AudioSegment.silent(duration=int(total_duration_ms))

    interval_ms = interval_seconds * 1000.0
    num_events = int(total_duration_ms / interval_ms) + 1

    if verbose:
        print(f"    Feedback: {len(samples)} samples, ~{num_events} events (random selection, overlapping)")

    gliss_count = 0
    total_played = 0

    for i in range(num_events):
        start_ms = i * interval_ms

        if start_ms >= total_duration_ms:
            break

        # Random sample selection
        sample_data = random.choice(samples)
        sample_pcs = sample_data["pitch_classes"]
        audio_path = sample_data["audio_path"]

        try:
            audio, gliss_log = render_sample_with_live_transposition(
                audio_path=audio_path,
                sample_pcs=sample_pcs,
                start_ms=start_ms,
                harmonic_timeline=harmonic_timeline,
                verbose=False,
                sample_name=sample_data["name"]
            )

            if len(audio) == 0:
                continue

            gliss_count += len(gliss_log)
            total_played += 1

            position_ms = int(start_ms)
            if position_ms + len(audio) > len(layer):
                extra = (position_ms + len(audio)) - len(layer)
                layer = layer + AudioSegment.silent(duration=int(extra))

            layer = layer.overlay(audio, position=position_ms)

        except Exception as e:
            if verbose:
                print(f"      Error loading {sample_data['name']}: {e}")

    if verbose:
        print(f"    Feedback: {total_played} samples played, {gliss_count} glissandos")

    return layer


def render_laken_layer(
    harmonic_timeline: List[HarmonicEvent],
    total_duration_ms: float,
    min_silence_seconds: float = 7.0,
    max_silence_seconds: float = 9.0,
    cloud_duration_seconds: float = 5.0,
    bpm: float = 120.0,
    verbose: bool = True,
    seed: Optional[int] = None
) -> AudioSegment:
    """
    Render Laken samples as sporadic clouds of 16th notes.

    Clouds emerge after random stretches of silence (7-9 seconds).
    Each cloud lasts about 5 seconds of continuous 16th notes.
    Each note uses REACTIVE transposition to fit the current chord.
    """
    if seed is not None:
        random.seed(seed + 888)  # Different seed offset for variety

    samples = load_laken_samples()
    if not samples:
        return AudioSegment.silent(duration=int(total_duration_ms))

    layer = AudioSegment.silent(duration=int(total_duration_ms))

    # 16th note interval at given BPM
    beat_ms = 60000.0 / bpm  # Quarter note
    sixteenth_ms = beat_ms / 4  # 16th note (125ms at 120 BPM)

    # Calculate notes per cloud based on cloud duration
    cloud_duration_ms = cloud_duration_seconds * 1000.0
    notes_per_cloud = int(cloud_duration_ms / sixteenth_ms)  # ~40 notes for 5 seconds

    if verbose:
        print(f"    Laken: {len(samples)} samples")
        print(f"    Clouds: ~{notes_per_cloud} 16th notes ({cloud_duration_seconds}s) after {min_silence_seconds}-{max_silence_seconds}s silence")
        print(f"    (16th note = {sixteenth_ms:.0f}ms at {bpm} BPM)")

    gliss_count = 0
    total_notes = 0
    num_clouds = 0
    next_sample_idx = 0

    # Start with a random silence period
    current_ms = random.uniform(min_silence_seconds, max_silence_seconds) * 1000.0

    while current_ms < total_duration_ms:
        # This is the start of a cloud
        cloud_start_ms = current_ms
        num_clouds += 1

        # Render notes_per_cloud 16th notes in this cloud
        for note_idx in range(notes_per_cloud):
            note_start_ms = cloud_start_ms + (note_idx * sixteenth_ms)

            if note_start_ms >= total_duration_ms:
                break

            # Cycle through samples
            sample_data = samples[next_sample_idx % len(samples)]
            next_sample_idx += 1

            sample_pcs = sample_data["pitch_classes"]
            audio_path = sample_data["audio_path"]

            try:
                audio, gliss_log = render_sample_with_live_transposition(
                    audio_path=audio_path,
                    sample_pcs=sample_pcs,
                    start_ms=note_start_ms,
                    harmonic_timeline=harmonic_timeline,
                    verbose=False,
                    sample_name=sample_data["name"]
                )

                if len(audio) == 0:
                    continue

                gliss_count += len(gliss_log)
                total_notes += 1

                position_ms = int(note_start_ms)
                if position_ms + len(audio) > len(layer):
                    extra = (position_ms + len(audio)) - len(layer)
                    layer = layer + AudioSegment.silent(duration=int(extra))

                layer = layer.overlay(audio, position=position_ms)

            except Exception as e:
                if verbose:
                    print(f"      Error loading {sample_data['name']}: {e}")

        # Move past the cloud, then add random silence before next cloud
        current_ms = cloud_start_ms + cloud_duration_ms
        silence_duration = random.uniform(min_silence_seconds, max_silence_seconds) * 1000.0
        current_ms += silence_duration

    if verbose:
        print(f"    Laken: {total_notes} notes played in {num_clouds} clouds, {gliss_count} glissandos")

    return layer


def get_harmonic_state_at_time(
    time_ms: float,
    harmonic_timeline: List[HarmonicEvent]
) -> Optional[HarmonicEvent]:
    """Get the harmonic event active at a given time.

    Returns the event with the LATEST start_ms that is <= time_ms.
    This handles overlapping events correctly (due to end_ms extensions).
    """
    # Find the most recent event that has started by this time
    best_event = None
    for he in harmonic_timeline:
        if he.start_ms <= time_ms:
            if best_event is None or he.start_ms > best_event.start_ms:
                best_event = he
    return best_event


def render_overlay_with_reactive_transposition(
    audio: AudioSegment,
    original_pcs: Set[int],
    start_ms: float,
    harmonic_timeline: List[HarmonicEvent],
    find_transposition_fn,
    verbose: bool = True,
    sample_name: str = "",
    debug: bool = False
) -> Tuple[AudioSegment, float, int]:
    """
    Render an overlay sample with EVENT-DRIVEN reactive transposition (like SNAPS).

    Architecture:
    1. At sample start, get active harmonic event and initial transposition
    2. Precompute harmonic change events within the output window
    3. Convert event times to output sample indices
    4. During rendering, check scheduled events (not polling every sample)
    5. On event: retarget glide from current interpolated position to new target
    6. Retargetable glides allowed mid-glide (like SNAPS voice behavior)

    Returns (processed_audio, output_duration_ms, gliss_count)
    """
    # Transposition bounds (like SNAPS: -9 to +5)
    MIN_TRANSPOSITION = -12
    MAX_TRANSPOSITION = 12

    original_rate = audio.frame_rate
    channels = audio.channels
    sample_width = audio.sample_width

    # Determine max value based on sample width for normalization
    if sample_width == 1:
        max_val = 127.0
    elif sample_width == 2:
        max_val = 32767.0
    elif sample_width == 3:
        max_val = 8388607.0
    elif sample_width == 4:
        max_val = 2147483647.0
    else:
        max_val = 32767.0  # fallback to 16-bit

    # Convert to numpy for processing
    raw_samples = audio.get_array_of_samples()
    if len(raw_samples) == 0:
        return AudioSegment.empty(), 0, 0

    # Normalize to -1.0 to 1.0 range for consistent processing
    samples = np.array(raw_samples, dtype=np.float32) / max_val
    if channels == 2:
        if len(samples) % 2 != 0:
            samples = samples[:-1]
        samples = samples.reshape((-1, 2))
    else:
        samples = samples.reshape((-1, 1))

    num_input_samples = len(samples)
    if num_input_samples < 2:
        return AudioSegment.empty(), 0, 0

    # Build harmonic index for fast lookup
    harmonic_start_times, harmonic_events_sorted = build_harmonic_index(harmonic_timeline)

    # === STEP 1: Get initial transposition at start_ms ===
    he = get_harmonic_event_at(harmonic_start_times, harmonic_events_sorted, start_ms)
    if he is None:
        initial_trans = 0
    else:
        initial_trans = find_transposition_fn(original_pcs, he.chord_pcs, he.chord_root)

    if initial_trans is None:
        # Can't fit at all - return empty
        if verbose:
            print(f"      ↳ dropped immediately (no valid transposition at start)")
        return AudioSegment.empty(), 0, 0

    # === STEP 2: Precompute scheduled harmonic change events ===
    # Estimate max output duration (if playing slow at rate 0.5)
    max_output_duration_ms = (num_input_samples / original_rate) * 1000 * 2
    end_time_ms = start_ms + max_output_duration_ms

    # scheduled_events: list of (output_sample_idx, target_transposition, chord_name)
    # These are the harmonic changes that occur DURING this sample's playback
    scheduled_events = []
    gliss_count = 0
    prev_trans = initial_trans
    dropout_idx = None

    for t, he in zip(harmonic_start_times, harmonic_events_sorted):
        if t <= start_ms:
            continue
        if t > end_time_ms:
            break

        trans = find_transposition_fn(original_pcs, he.chord_pcs, he.chord_root)

        # Convert event time to output sample index
        # output_sample_idx = (event_time_ms - start_ms) * sample_rate / 1000
        event_output_idx = int((t - start_ms) * original_rate / 1000)

        if trans is None:
            # Mark dropout point
            dropout_idx = event_output_idx
            scheduled_events.append((event_output_idx, None, he.chord_name))
            break

        if trans != prev_trans:
            scheduled_events.append((event_output_idx, trans, he.chord_name))
            gliss_count += 1
            prev_trans = trans

    # Debug logging for first Organetta sample (or any sample with debug=True)
    is_first_organetta = "Organetta" in sample_name and start_ms < 5000
    if debug or is_first_organetta:
        print(f"\n      === EVENT-DRIVEN DEBUG: {sample_name} ===")
        print(f"      start_ms={start_ms:.0f}, original_pcs={sorted(original_pcs)}")
        print(f"      input_samples={num_input_samples}, est_duration={num_input_samples/original_rate*1000:.0f}ms")
        if he:
            print(f"      initial chord: {he.chord_name}, pcs={sorted(he.chord_pcs)}")
        print(f"      initial_trans={initial_trans}")
        print(f"      scheduled_events ({len(scheduled_events)}):")
        for idx, trans, chord in scheduled_events:
            print(f"        @sample {idx} ({idx*1000/original_rate:.0f}ms): {chord} -> trans={trans}")

    # === STEP 3: Render with event-driven glide retargeting ===
    gliss_duration_ms = GLISSANDO_MS
    gliss_samples = int(gliss_duration_ms * original_rate / 1000)
    max_output_samples = int(num_input_samples / 0.5) + original_rate

    # Pre-allocate output
    output = np.zeros((max_output_samples, channels), dtype=np.float32)

    input_pos = 0.0
    output_idx = 0
    event_idx = 0  # Index into scheduled_events

    # Glide state - initialized before loop
    current_trans = float(initial_trans)
    target_trans = float(initial_trans)
    gliss_start_trans = float(initial_trans)
    gliss_end_trans = float(initial_trans)
    gliss_start_output_idx = 0
    in_gliss = False

    trans_log = []

    while input_pos < num_input_samples - 1 and output_idx < max_output_samples:
        # === Check if we've reached a scheduled event ===
        while event_idx < len(scheduled_events) and output_idx >= scheduled_events[event_idx][0]:
            event_output_idx, new_trans, chord_name = scheduled_events[event_idx]
            event_idx += 1

            if new_trans is None:
                # Dropout - fade out and stop
                fade_len = min(int(0.1 * original_rate), max_output_samples - output_idx)
                for fi in range(fade_len):
                    fade_gain = 1.0 - (fi / fade_len)
                    int_pos = int(input_pos)
                    frac = input_pos - int_pos
                    if int_pos + 1 < num_input_samples:
                        output[output_idx] = (samples[int_pos] * (1 - frac) + samples[int_pos + 1] * frac) * fade_gain
                    input_pos += 1.0
                    output_idx += 1
                # Break out of main loop
                input_pos = num_input_samples
                break

            # === RETARGETABLE GLIDE (like SNAPS) ===
            # Start new glide from CURRENT interpolated position to new target
            gliss_start_trans = current_trans  # Where we ARE right now
            gliss_end_trans = float(new_trans)
            gliss_start_output_idx = output_idx
            in_gliss = True
            target_trans = float(new_trans)
            trans_log.append((output_idx * 1000 / original_rate, current_trans, new_trans))

        if input_pos >= num_input_samples - 1:
            break

        # === Calculate current transposition (with glide interpolation) ===
        if in_gliss:
            gliss_progress = (output_idx - gliss_start_output_idx) / gliss_samples
            if gliss_progress >= 1.0:
                # Glide complete
                in_gliss = False
                current_trans = gliss_end_trans
            else:
                # Quarter-sine easing (like SNAPS SCurve)
                eased = np.sin(gliss_progress * np.pi / 2)
                current_trans = gliss_start_trans + (gliss_end_trans - gliss_start_trans) * eased

        # === Sanity clamp transposition ===
        current_trans = max(MIN_TRANSPOSITION, min(MAX_TRANSPOSITION, current_trans))

        # === Calculate playback rate ===
        rate = 2 ** (current_trans / 12.0)

        # === Interpolate sample at current input position ===
        int_pos = int(input_pos)
        frac = input_pos - int_pos

        if int_pos + 1 < num_input_samples:
            output[output_idx] = samples[int_pos] * (1 - frac) + samples[int_pos + 1] * frac
        elif int_pos < num_input_samples:
            output[output_idx] = samples[int_pos]
        else:
            break

        input_pos += rate
        output_idx += 1

    # Trim output
    output = output[:output_idx]

    if debug:
        print(f"      output_samples={len(output)}, duration={len(output)/original_rate*1000:.0f}ms")
        if len(output) > 0:
            print(f"      output max_abs={np.max(np.abs(output)):.1f}, min={np.min(output):.1f}, max={np.max(output):.1f}")
        print(f"      === END DEBUG ===\n")

    if verbose and trans_log:
        for t, from_t, to_t in trans_log:
            print(f"      ↳ glissando at {t/1000:.2f}s: {from_t:+.1f} → {to_t:+.1f} semitones")

    if len(output) == 0:
        return AudioSegment.empty(), 0, gliss_count

    # Convert back to AudioSegment (always output as 16-bit for consistency)
    output_flat = output.flatten()
    if channels == 2 and len(output_flat) % 2 != 0:
        output_flat = output_flat[:-1]

    # Scale from normalized (-1.0 to 1.0) back to 16-bit range
    output_int16 = np.clip(output_flat * 32767.0, -32768, 32767).astype(np.int16)

    try:
        result = AudioSegment(
            data=output_int16.tobytes(),
            sample_width=2,  # Always output 16-bit
            frame_rate=original_rate,
            channels=channels
        )
        return result, len(result), gliss_count
    except Exception:
        return AudioSegment.empty(), 0, gliss_count


def render_bassflute_with_live_transposition(
    event: BassFluteEvent,
    harmonic_timeline: List[HarmonicEvent],
    verbose: bool = True
) -> Tuple[AudioSegment, float]:
    """
    Render a bass flute sample with REACTIVE transposition.

    Simple approach: at each moment, check the harmonic state and transpose
    if needed. If the sample can't fit the current harmony, it fades out.

    Returns (processed_audio, output_duration_ms)
    """
    bassflute_samples = load_bassflute_samples()
    bf_data = bassflute_samples.get(event.sample_name)
    if not bf_data:
        return AudioSegment.empty(), 0

    audio_path = bf_data["audio_path"]
    if not audio_path.exists():
        return AudioSegment.empty(), 0

    original_audio = AudioSegment.from_file(audio_path)

    result, duration, gliss_count = render_overlay_with_reactive_transposition(
        audio=original_audio,
        original_pcs=event.original_pcs,
        start_ms=event.start_ms,
        harmonic_timeline=harmonic_timeline,
        find_transposition_fn=find_bassflute_transposition,
        verbose=verbose,
        sample_name=event.sample_name
    )
    return result, duration


def render_bassflute_layer(
    events: List[BassFluteEvent],
    harmonic_timeline: List[HarmonicEvent],
    total_duration_ms: float,
    verbose: bool = True
) -> AudioSegment:
    """
    Render bass flute one-shots with LIVE transposition.

    Samples play CONTINUOUSLY - when one ends, the next immediately starts.
    No gaps, no overlap. Each sample re-transposes with glissando
    whenever the harmonic state changes during its playback.
    """
    if not events:
        return AudioSegment.silent(duration=int(total_duration_ms))

    # Create silent base
    layer = AudioSegment.silent(duration=int(total_duration_ms))

    current_ms = 0.0  # Track actual position as we render
    played_count = 0

    for event in events:
        if current_ms >= total_duration_ms:
            break

        # Update the event's start_ms to the actual current position
        event.start_ms = current_ms

        if verbose:
            print(f"    {event.sample_name} starting at {event.start_ms/1000:.2f}s")

        audio, duration = render_bassflute_with_live_transposition(
            event, harmonic_timeline, verbose=verbose
        )

        if len(audio) == 0:
            # Skip ahead a bit to avoid infinite loop
            current_ms += 1000
            continue

        played_count += 1

        # Overlay at the correct position
        position_ms = int(current_ms)
        if position_ms < 0:
            position_ms = 0
        if position_ms + len(audio) > len(layer):
            extra = (position_ms + len(audio)) - len(layer)
            layer = layer + AudioSegment.silent(duration=int(extra))

        layer = layer.overlay(audio, position=position_ms)

        # Move to end of this sample (no gap)
        current_ms += len(audio)

    if verbose:
        print(f"    Bass flute: {played_count} samples played continuously")

    return layer


def render_chain_to_wav(
    chain: List[ChainLink],
    output_path: Path,
    verbose: bool = True,
    include_midi_chords: bool = True,
    include_bassflute: bool = True,
    bassflute_density: float = 0.4,
    include_brodero: bool = False,
    include_jicello: bool = False,
    include_organetta: bool = False,
    organetta_interval: float = 4.0,
    include_minorchordbeat: bool = False,
    include_mutebowl: bool = False,
    include_qualitychords: bool = False,
    qualitychords_density: float = 0.4,
    # NOTE: include_progressions removed - progressions are skeleton samples in chain pool,
    # not an overlay layer. Control via load_all_samples(include_progressions=...) instead.
    include_prophetfalse: bool = False,
    prophetfalse_interval: float = 3.0,
    include_harmonicker: bool = False,
    harmonicker_interval: float = 5.0,
    include_gothicharp: bool = False,
    include_laken: bool = False,
    include_gentleharpsi: bool = False,
    include_feedback: bool = False,
    feedback_interval: float = 4.0,
    include_stylo: bool = False,
    stylo_interval: float = 2.0,
    include_trichords: bool = False,
    include_tremolo_oct: bool = False,
    tremolo_oct_interval: float = 4.0,
    include_averyviolin: bool = False,
    include_dictamel: bool = False,
    include_scelsipezzi: bool = False,
    scelsipezzi_interval: float = 8.0,
    include_godette: bool = False,
    include_synth_bass: bool = True,
    render_midi_synth: bool = True,
    seed: Optional[int] = None,
    max_duration_ms: Optional[int] = None
) -> Path:
    """
    Render the chain to a WAV file.

    If include_midi_chords is True, also renders the inferred chords as MIDI
    starting at the halfway point of each sample (when last_collection plays).

    If include_bassflute is True, layers in bass flute one-shots that transpose
    to fit the current harmonic state with 150ms glissando.

    If include_brodero is True, layers rapid-fire Brodero samples in alphanumeric
    order at the specified rate.
    """
    if verbose:
        print("\nRendering audio...")

    # === TIMING INSTRUMENTATION ===
    layer_timings = {}

    # === NUMPY BUFFER APPROACH ===
    # First pass: render all chain samples and track ACTUAL rendered timing
    # Audio render creates the clock. MIDI follows that clock. Not the other way around.
    # NOTE: All samples are resampled to TARGET_SAMPLE_RATE (44100) on load via
    # get_cached_audio(), so timing calculations use this consistent rate.
    SAMPLE_RATE = TARGET_SAMPLE_RATE
    CHANNELS = 2
    chain_samples_list = []  # [(samples_np, duration_samples)]
    rendered_events = []  # List[RenderedSkeletonEvent] - actual timing
    rendered_cursor_ms = 0.0  # Single source of truth for timing

    for i, link in enumerate(chain):
        sample_audio_dir = link.audio_dir if link.audio_dir else AUDIO_DIR
        sample_filename = link.sample
        if sample_filename.startswith("handel_"):
            sample_filename = sample_filename[7:]
        elif sample_filename.startswith("glaz_"):
            sample_filename = sample_filename[5:]
        elif sample_filename.startswith("hyacinthe_"):
            sample_filename = sample_filename[10:]
        elif sample_filename.startswith("kraus_"):
            sample_filename = sample_filename[6:]
        audio_path = sample_audio_dir / f"{sample_filename}.wav"

        if not audio_path.exists():
            print(f"  WARNING: {audio_path} not found")
            continue

        # Use cached audio loading
        cached = get_cached_audio(audio_path)
        if cached is None:
            continue

        samples = cached["samples"].copy()
        sr = cached["sample_rate"]

        # Apply transposition if needed
        if link.transposition != 0:
            samples = apply_varispeed_np(samples, float(link.transposition))

        # Calculate ACTUAL rendered duration from the samples
        # NOTE: Use SAMPLE_RATE (44100) because pydub plays audio at this rate.
        # The 48kHz samples get stretched to 44.1kHz during playback, so
        # timing calculations must match that stretched playback.
        rendered_duration_ms = len(samples) * 1000.0 / SAMPLE_RATE
        rendered_start_ms = rendered_cursor_ms
        rendered_end_ms = rendered_start_ms + rendered_duration_ms

        # Calculate chord timing in RENDERED space (not source space)
        # chord_b_onset_ratio is a ratio of duration, so it scales with varispeed
        chord_a_start_ms = rendered_start_ms
        chord_b_start_ms = rendered_start_ms + (rendered_duration_ms * link.chord_b_onset_ratio)

        # Create RenderedSkeletonEvent with actual timing
        rendered_event = RenderedSkeletonEvent(
            sample_name=link.sample,
            transposition=link.transposition,
            source_duration_ms=link.duration_ms,
            rendered_start_ms=rendered_start_ms,
            rendered_duration_ms=rendered_duration_ms,
            rendered_end_ms=rendered_end_ms,
            first_pcs=link.first_pcs,
            last_pcs=link.last_pcs,
            inferred_chord=link.inferred_chord,
            chord_b_onset_ratio=link.chord_b_onset_ratio,
            chord_a_start_ms=chord_a_start_ms,
            chord_b_start_ms=chord_b_start_ms
        )
        rendered_events.append(rendered_event)

        # Normalize (simple peak normalization)
        # Samples are in [-1.0, 1.0] range, so target_peak is also normalized
        peak = np.max(np.abs(samples))
        if peak > 0:
            target_peak = apply_gain_db(SAMPLE_NORMALIZE_DB)  # e.g., -5dB -> ~0.56
            samples = samples * (target_peak / peak)

        # Ensure stereo
        if samples.shape[1] == 1:
            samples = np.hstack([samples, samples])
        elif samples.shape[1] > 2:
            samples = samples[:, :2]

        chain_samples_list.append((samples, len(samples)))

        # Advance the rendered cursor by ACTUAL duration
        rendered_cursor_ms = rendered_end_ms

        if verbose:
            trans_str = f" (trans {link.transposition:+d})" if link.transposition else ""
            print(f"  {link.sample}{trans_str}: {int(rendered_duration_ms)}ms")

        # Add a gap between skeleton samples (5-10 seconds of silence)
        # This allows sustained layers to breathe and hold harmonic states longer
        if i < len(chain) - 1:  # Don't add gap after last sample
            gap_ms = random.uniform(5000, 10000)  # 5-10 seconds
            gap_samples = int(gap_ms * SAMPLE_RATE / 1000)
            silence = np.zeros((gap_samples, CHANNELS), dtype=np.float32)
            chain_samples_list.append((silence, gap_samples))
            rendered_cursor_ms += gap_ms
            if verbose:
                print(f"    [gap: {gap_ms/1000:.1f}s]")

    # Calculate total samples
    total_chain_samples = sum(num_samples for _, num_samples in chain_samples_list)

    # Create master buffer and concatenate chain samples
    master_buffer = np.zeros((total_chain_samples, CHANNELS), dtype=np.float32)
    current_sample = 0
    for samples, num_samples in chain_samples_list:
        master_buffer[current_sample:current_sample + num_samples] = samples
        current_sample += num_samples

    current_time_ms = rendered_cursor_ms  # Use the rendered cursor as total time
    chain_duration_ms = current_time_ms

    # Limit layer duration to requested max_duration_ms
    if max_duration_ms is not None:
        total_duration_ms = min(chain_duration_ms, float(max_duration_ms))
    else:
        total_duration_ms = chain_duration_ms

    if verbose:
        print(f"  Chain rendered to numpy buffer: {total_chain_samples} samples ({chain_duration_ms/1000:.2f}s)")
        if max_duration_ms is not None:
            print(f"  Layer duration limited to: {total_duration_ms/1000:.2f}s (requested: {max_duration_ms/1000:.2f}s)")

    # === BUILD HARMONIC TIMELINE FROM RENDERED TIMING ===
    # This ensures MIDI uses the exact same timing as the audio render
    harmonic_timeline = build_harmonic_timeline_from_rendered(rendered_events, chain)
    if verbose:
        print(f"  Built harmonic timeline with {len(harmonic_timeline)} chord events (from rendered timing)")

    # Build harmonic index for fast O(log n) lookup
    harmonic_start_times, harmonic_events_sorted = build_harmonic_index(harmonic_timeline)

    # Convert master_buffer to AudioSegment for compatibility with existing layer code
    # This is a temporary bridge while we migrate layers to numpy
    combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

    # Convert harmonic timeline to chord_events format for MIDI generation
    # This ensures MIDI uses the exact same timing as everything else
    chord_events = []
    for i, he in enumerate(harmonic_timeline):
        # Calculate duration from this event to the next (or end)
        if i < len(harmonic_timeline) - 1:
            duration_ms = harmonic_timeline[i + 1].start_ms - he.start_ms
        else:
            # Last event: sustain until end of audio
            duration_ms = current_time_ms - he.start_ms
            if duration_ms < 0:
                duration_ms = 5000  # Fallback: 5 seconds

        chord_events.append({
            "start_ms": he.start_ms,
            "duration_ms": duration_ms,
            "chord": {
                "name": he.chord_name,
                "pitch_classes": list(he.chord_pcs),
                "root": he.chord_root,
            },
        })

    # Generate MIDI files (exported separately, not rendered to audio)
    voicing_midi_path = None
    bass_midi_path = None

    if include_midi_chords and chord_events:
        if verbose:
            print("\nGenerating MIDI files...")

        # Get the chord dictionary for original voicings
        chord_dict = load_chord_dictionary()

        # Create separate MIDI files for voicing and bass
        import pretty_midi

        # Voicing MIDI (keys/piano)
        voicing_midi = pretty_midi.PrettyMIDI()
        keys = pretty_midi.Instrument(program=0, name="Keys")

        # Bass MIDI (sub bass roots) - eighth note stabs synced with beat
        bass_midi = pretty_midi.PrettyMIDI()
        sub_bass = pretty_midi.Instrument(program=38, name="SubBass")

        # Build a lookup for which chord is active at any given time
        # chord_events is sorted by start_ms
        def get_root_at_time(time_ms):
            """Get the chord root pitch class active at a given time."""
            for i, event in enumerate(chord_events):
                event_start = event["start_ms"]
                event_end = event_start + event["duration_ms"]
                if event_start <= time_ms < event_end:
                    chord_data = chord_dict.get(event["chord"]["name"], {})
                    return chord_data.get("root", 0)
            # If past all events, use the last chord's root
            if chord_events:
                chord_data = chord_dict.get(chord_events[-1]["chord"]["name"], {})
                return chord_data.get("root", 0)
            return 0

        # Generate bass notes aligned with chord events (same onset as voicing)
        for event in chord_events:
            root_pc = get_root_at_time(event["start_ms"])
            root_midi = 36 + root_pc  # C2 range for deep sub bass

            start_sec = event["start_ms"] / 1000.0
            duration_sec = event["duration_ms"] / 1000.0

            bass_note = pretty_midi.Note(
                velocity=110,  # Punchy
                pitch=root_midi,
                start=start_sec,
                end=start_sec + duration_sec
            )
            sub_bass.notes.append(bass_note)

        if verbose:
            print(f"  Bass: {len(chord_events)} sustained notes (aligned with voicing chords)")

        # Add voicing notes to keys MIDI (sustained, for export only)
        for event in chord_events:
            chord_name = event["chord"]["name"]
            chord_data = chord_dict.get(chord_name, {})
            voicing = chord_data.get("original_voicing", [])

            # If no voicing found (synthetic chord), generate one from pitch classes
            if not voicing:
                pcs = event["chord"].get("pitch_classes", [])
                if pcs:
                    # Generate voicing: root in bass (octave 3), rest spread in octave 4-5
                    root = event["chord"].get("root", min(pcs) if pcs else 0)
                    voicing = [48 + root]  # Root in octave 3
                    for pc in sorted(pcs):
                        if pc != root:
                            # Place non-root notes in octave 4 or 5
                            voicing.append(60 + pc if pc < 6 else 60 + pc)

            start_sec = event["start_ms"] / 1000.0
            duration_sec = event["duration_ms"] / 1000.0

            for midi_note in voicing:
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=midi_note,
                    start=start_sec,
                    end=start_sec + duration_sec
                )
                keys.notes.append(note)

        voicing_midi.instruments.append(keys)
        bass_midi.instruments.append(sub_bass)

        # Export MIDI files (paths will be set after we know output_path)
        # Store for later export
        voicing_midi_obj = voicing_midi
        bass_midi_obj = bass_midi

        if verbose:
            print(f"  Generated {len(chord_events)} chord events for MIDI export")

        # Render MIDI to audio and mix in (only if synth rendering is enabled)
        if render_midi_synth:
            if verbose:
                print("  Rendering MIDI synth to audio...")

            # Render sub bass to audio (if enabled)
            if include_synth_bass:
                bass_only_midi = pretty_midi.PrettyMIDI()
                bass_only_midi.instruments.append(sub_bass)
                bass_audio = render_midi_to_audio(bass_only_midi)

            # Render voicing (keys) to audio - piano-like sound
            keys_only_midi = pretty_midi.PrettyMIDI()
            keys_only_midi.instruments.append(keys)
            keys_audio = render_midi_to_audio(keys_only_midi)

            # Ensure same length as combined audio
            if include_synth_bass:
                if len(bass_audio) < len(combined):
                    bass_audio = bass_audio + AudioSegment.silent(duration=len(combined) - len(bass_audio))
                elif len(bass_audio) > len(combined):
                    bass_audio = bass_audio[:len(combined)]

            if len(keys_audio) < len(combined):
                keys_audio = keys_audio + AudioSegment.silent(duration=len(combined) - len(keys_audio))
            elif len(keys_audio) > len(combined):
                keys_audio = keys_audio[:len(combined)]

            # Mix MIDI layers (boost keys to be clearly audible)
            if include_synth_bass:
                bass_audio = bass_audio - 3  # -3 dB for bass
                combined = combined.overlay(bass_audio)
            keys_audio = keys_audio + 3  # +3 dB for piano voicing (louder, clearly audible)
            combined = combined.overlay(keys_audio)

            if verbose:
                if include_synth_bass:
                    print(f"  Mixed MIDI synth layers into audio (bass + piano voicing)")
                else:
                    print(f"  Mixed MIDI synth layers into audio (piano voicing only, bass skipped)")
        else:
            if verbose:
                print("  MIDI synth rendering disabled (files will still be exported)")

    # Layer in bass flute one-shots - NUMPY VERSION
    if verbose:
        print("\n>>> START layer: bass_flute")
    t_bassflute_start = time.perf_counter()
    if include_bassflute:
        if verbose:
            print("\nLayering bass flute one-shots (NumPy engine)...")

        bassflute_events = select_bassflute_events(
            chain,
            harmonic_timeline=harmonic_timeline,
            density=bassflute_density,
            seed=seed,
            total_duration_ms=total_duration_ms
        )

        if bassflute_events:
            if verbose:
                print(f"  Queued {len(bassflute_events)} bass flute samples for continuous playback")

            # Load bassflute sample metadata
            bf_samples_dict = load_bassflute_samples()

            # Convert events to sample_list format for numpy renderer
            bf_sample_list = []
            for event in bassflute_events:
                sample_data = bf_samples_dict.get(event.sample_name)
                if sample_data:
                    bf_sample_list.append({
                        "name": event.sample_name,
                        "pitch_classes": sample_data["pitch_classes"],
                        "audio_path": sample_data["audio_path"]
                    })

            if bf_sample_list:
                # Use unified rendering with LayerConfig
                bassflute_config = LayerConfig(
                    name="Bass flute",
                    samples=bf_sample_list,
                    layer_type=LayerType.CONTINUOUS,
                    gain_db=+3.0,
                )

                bassflute_np = render_layer(
                    config=bassflute_config,
                    harmonic_start_times=harmonic_start_times,
                    harmonic_events=harmonic_events_sorted,
                    find_transposition_fn=find_bassflute_transposition,
                    total_duration_ms=total_duration_ms,
                    sample_rate=SAMPLE_RATE,
                    channels=CHANNELS,
                    verbose=verbose,
                    seed=seed,
                    render_continuous_fn=render_layer_continuous_np,
                    render_interval_fn=render_layer_interval_np,
                )

                usable_len = min(len(bassflute_np), len(master_buffer))
                master_buffer[:usable_len] += bassflute_np[:usable_len]
                combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

                if verbose:
                    print(f"  Mixed bass flute layer (NumPy)")
        else:
            if verbose:
                print("  No suitable bass flute samples found for current chords")
    layer_timings['bass_flute'] = time.perf_counter() - t_bassflute_start
    if verbose:
        print(f"<<< END layer: bass_flute ({layer_timings['bass_flute']:.2f}s)")

    # Layer in Brodero samples (clouds of 16th notes with silence) - UNIFIED NUMPY VERSION
    if verbose:
        print("\n>>> START layer: brodero")
    t_brodero_start = time.perf_counter()
    if include_brodero:
        if verbose:
            print("\nLayering Brodero samples (NumPy engine, 16th note clouds)...")

        brodero_samples = load_brodero_samples()
        if brodero_samples:
            brodero_config = LayerConfig(
                name="Brodero",
                samples=brodero_samples,
                layer_type=LayerType.CLOUD,
                min_silence_seconds=10.0,
                max_silence_seconds=15.0,
                cloud_duration_seconds=4.0,
                bpm=120.0,
                note_division=16,
                gain_db=-15.0,
                selection=SelectionMode.SEQUENTIAL,
            )

            brodero_np = render_layer(
                config=brodero_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_continuous_fn=render_layer_continuous_np,
                render_interval_fn=render_layer_interval_np,
                render_cloud_fn=render_layer_cloud_np,
            )

            usable_len = min(len(brodero_np), len(master_buffer))
            master_buffer[:usable_len] += brodero_np[:usable_len]
            combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

            if verbose:
                print(f"  Mixed Brodero layer (NumPy)")

    # Layer in Jicello samples (distributed evenly) - UNIFIED NUMPY VERSION
    if include_jicello:
        if verbose:
            print("\nLayering Jicello samples CONTINUOUSLY (NumPy engine)...")

        jicello_samples = load_jicello_samples()
        if jicello_samples:
            jicello_config = LayerConfig(
                name="Jicello",
                samples=jicello_samples,
                layer_type=LayerType.CONTINUOUS,
                gain_db=-12.0,
            )

            jicello_np = render_layer(
                config=jicello_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_continuous_fn=render_layer_continuous_np,
                render_interval_fn=render_layer_interval_np,
            )

            usable_len = min(len(jicello_np), len(master_buffer))
            master_buffer[:usable_len] += jicello_np[:usable_len]
            combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

            if verbose:
                print(f"  Mixed Jicello layer (NumPy)")

    # Layer in Organetta samples (every N seconds) - UNIFIED NUMPY VERSION
    if verbose:
        print("\n>>> START layer: organetta")
    t_organetta_start = time.perf_counter()
    if include_organetta:
        if verbose:
            print("\nLayering Organetta samples (NumPy engine)...")

        organetta_samples = load_organetta_samples()
        if organetta_samples:
            organetta_config = LayerConfig(
                name="Organetta",
                samples=organetta_samples,
                layer_type=LayerType.INTERVAL,
                interval_seconds=organetta_interval,
                gain_db=-18.0,
                selection=SelectionMode.SEQUENTIAL,
            )

            organetta_np = render_layer(
                config=organetta_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_continuous_fn=render_layer_continuous_np,
                render_interval_fn=render_layer_interval_np,
            )

            usable_len = min(len(organetta_np), len(master_buffer))
            master_buffer[:usable_len] += organetta_np[:usable_len]
            combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

            if verbose:
                print(f"  Mixed Organetta layer (NumPy)")
    layer_timings['organetta'] = time.perf_counter() - t_organetta_start
    if verbose:
        print(f"<<< END layer: organetta ({layer_timings['organetta']:.2f}s)")

    # Layer in MinorChordBeat samples (eighth notes) - UNIFIED NUMPY VERSION
    if include_minorchordbeat:
        if verbose:
            print("\nLayering MinorChordBeat samples (NumPy engine, eighth notes @ 120 BPM)...")

        minorchordbeat_samples = load_minorchordbeat_samples()
        if minorchordbeat_samples:
            # Eighth note at 120 BPM = 0.5 seconds
            minorchordbeat_config = LayerConfig(
                name="MinorChordBeat",
                samples=minorchordbeat_samples,
                layer_type=LayerType.INTERVAL,
                interval_seconds=0.5,  # 120 BPM eighth note
                gain_db=-6.0,
                selection=SelectionMode.SEQUENTIAL,
            )

            minorchordbeat_np = render_layer(
                config=minorchordbeat_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_continuous_fn=render_layer_continuous_np,
                render_interval_fn=render_layer_interval_np,
            )

            usable_len = min(len(minorchordbeat_np), len(master_buffer))
            master_buffer[:usable_len] += minorchordbeat_np[:usable_len]
            combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

            if verbose:
                print(f"  Mixed MinorChordBeat layer (NumPy)")

    # Layer in MuteBowl samples (eighth notes) - UNIFIED NUMPY VERSION
    if include_mutebowl:
        if verbose:
            print("\nLayering MuteBowl samples (NumPy engine, eighth notes @ 120 BPM)...")

        mutebowl_samples = load_mutebowl_samples()
        if mutebowl_samples:
            mutebowl_config = LayerConfig(
                name="MuteBowl",
                samples=mutebowl_samples,
                layer_type=LayerType.INTERVAL,
                interval_seconds=0.5,  # 120 BPM eighth note
                gain_db=-6.0,
                selection=SelectionMode.SEQUENTIAL,
            )

            mutebowl_np = render_layer(
                config=mutebowl_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_continuous_fn=render_layer_continuous_np,
                render_interval_fn=render_layer_interval_np,
            )

            usable_len = min(len(mutebowl_np), len(master_buffer))
            master_buffer[:usable_len] += mutebowl_np[:usable_len]
            combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

            if verbose:
                print(f"  Mixed MuteBowl layer (NumPy)")

    # Layer in quality-aware chord samples (major/minor, sporadic)
    # (harmonic_timeline already built at start of function)
    if include_qualitychords:
        if verbose:
            print("\nLayering QualityChords (major/minor, sporadic)...")

        qualitychords_layer = render_qualitychords_layer(
            harmonic_timeline=harmonic_timeline,
            total_duration_ms=len(combined),
            bpm=120.0,
            sparsity=qualitychords_density,
            seed=seed,
            verbose=verbose
        )

        # Ensure same length
        if len(qualitychords_layer) < len(combined):
            qualitychords_layer = qualitychords_layer + AudioSegment.silent(
                duration=len(combined) - len(qualitychords_layer)
            )
        elif len(qualitychords_layer) > len(combined):
            qualitychords_layer = qualitychords_layer[:len(combined)]

        # Mix QualityChords (moderate volume, normalized samples)
        qualitychords_layer = qualitychords_layer - 6  # -6 dB
        combined = combined.overlay(qualitychords_layer)

        if verbose:
            print(f"  Mixed QualityChords layer with chain")

    # NOTE: Progressions (glaz_sax, hyacinthe, kraus) are skeleton samples.
    # They participate in chain building and are concatenated sequentially.
    # There is NO progressions overlay layer - that concept has been removed.
    # The --progressions flag controls whether progression samples are in the chain pool.

    # Layer in Prophet False samples (synth one-shots) - UNIFIED NUMPY VERSION
    if include_prophetfalse:
        if verbose:
            print("\nLayering Prophet False samples (NumPy engine)...")

        prophetfalse_samples = load_prophetfalse_samples()
        if prophetfalse_samples:
            prophetfalse_config = LayerConfig(
                name="ProphetFalse",
                samples=prophetfalse_samples,
                layer_type=LayerType.INTERVAL,
                interval_seconds=prophetfalse_interval,
                gain_db=-6.0,  # Was -9 dB, now louder
                selection=SelectionMode.SEQUENTIAL,
            )

            prophetfalse_np = render_layer(
                config=prophetfalse_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_continuous_fn=render_layer_continuous_np,
                render_interval_fn=render_layer_interval_np,
            )

            # Mix into master buffer (NumPy)
            usable_len = min(len(prophetfalse_np), len(master_buffer))
            master_buffer[:usable_len] += prophetfalse_np[:usable_len]

            # Update combined AudioSegment for compatibility
            combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

            if verbose:
                print(f"  Mixed Prophet False layer (NumPy)")

    # Layer in Harmonicker samples (harmonica chords) - UNIFIED NUMPY VERSION
    if include_harmonicker:
        if verbose:
            print("\nLayering Harmonicker samples (NumPy engine)...")

        harmonicker_samples = load_harmonicker_samples()
        if harmonicker_samples:
            harmonicker_config = LayerConfig(
                name="Harmonicker",
                samples=harmonicker_samples,
                layer_type=LayerType.INTERVAL,
                interval_seconds=harmonicker_interval,
                gain_db=-30.0,
                selection=SelectionMode.SEQUENTIAL,
            )

            harmonicker_np = render_layer(
                config=harmonicker_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_continuous_fn=render_layer_continuous_np,
                render_interval_fn=render_layer_interval_np,
            )

            # Mix into master buffer (NumPy)
            usable_len = min(len(harmonicker_np), len(master_buffer))
            master_buffer[:usable_len] += harmonicker_np[:usable_len]

            # Update combined AudioSegment for compatibility
            combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

            if verbose:
                print(f"  Mixed Harmonicker layer (NumPy)")

    # ==========================================================================
    # CLOUD LAYERS WITH DYNAMIC PANNING
    # Render all clouds first, then apply dynamic panning when they overlap
    # ==========================================================================
    cloud_layers = []  # List of (name, buffer, base_pan)

    # Layer in Gothic Harp clouds (sporadic 16th note bursts)
    if include_gothicharp:
        if verbose:
            print("\nLayering Gothic Harp clouds (NumPy engine, 16th note bursts)...")

        gothicharp_samples = load_gothicharp_samples()
        if gothicharp_samples:
            gothicharp_config = LayerConfig(
                name="Gothic Harp",
                samples=gothicharp_samples,
                layer_type=LayerType.CLOUD,
                min_silence_seconds=7.0,
                max_silence_seconds=9.0,
                cloud_duration_seconds=5.0,
                bpm=120.0,
                note_division=16,
                gain_db=-9.0,
                selection=SelectionMode.SEQUENTIAL,
            )

            gothicharp_np = render_layer(
                config=gothicharp_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_continuous_fn=render_layer_continuous_np,
                render_interval_fn=render_layer_interval_np,
                render_cloud_fn=render_layer_cloud_np,
            )
            # Base pan: slightly left (-0.3)
            cloud_layers.append(("Gothic Harp", gothicharp_np, -0.3))

    # Layer in Laken clouds (sporadic 16th note bursts)
    if include_laken:
        if verbose:
            print("\nLayering Laken clouds (NumPy engine, 16th note bursts)...")

        laken_samples = load_laken_samples()
        if laken_samples:
            laken_config = LayerConfig(
                name="Laken",
                samples=laken_samples,
                layer_type=LayerType.CLOUD,
                min_silence_seconds=7.0,
                max_silence_seconds=9.0,
                cloud_duration_seconds=5.0,
                bpm=120.0,
                note_division=16,
                gain_db=-15.0,
                selection=SelectionMode.SEQUENTIAL,
            )

            laken_np = render_layer(
                config=laken_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_continuous_fn=render_layer_continuous_np,
                render_interval_fn=render_layer_interval_np,
                render_cloud_fn=render_layer_cloud_np,
            )
            # Base pan: center (0.0)
            cloud_layers.append(("Laken", laken_np, 0.0))

    # Layer in Gentle Harpsichord clouds (sporadic 32nd note bursts)
    if include_gentleharpsi:
        if verbose:
            print("\nLayering Gentle Harpsichord clouds (NumPy engine, 32nd note bursts)...")

        gentleharpsi_samples = load_gentleharpsi_samples()
        if gentleharpsi_samples:
            gentleharpsi_config = LayerConfig(
                name="Gentle Harpsichord",
                samples=gentleharpsi_samples,
                layer_type=LayerType.CLOUD,
                min_silence_seconds=10.0,
                max_silence_seconds=11.0,
                cloud_duration_seconds=6.0,
                bpm=120.0,
                note_division=32,  # 32nd notes
                gain_db=-9.0,
                selection=SelectionMode.SEQUENTIAL,
            )

            gentleharpsi_np = render_layer(
                config=gentleharpsi_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_continuous_fn=render_layer_continuous_np,
                render_interval_fn=render_layer_interval_np,
                render_cloud_fn=render_layer_cloud_np,
            )
            # Base pan: slightly right (+0.3)
            cloud_layers.append(("Gentle Harpsichord", gentleharpsi_np, 0.3))

    # Apply dynamic panning to clouds based on overlap
    if cloud_layers:
        if verbose:
            print("\nApplying dynamic stereo panning to clouds...")

        # Detect activity for each cloud layer
        cloud_activities = []
        for name, buf, base_pan in cloud_layers:
            usable_len = min(len(buf), len(master_buffer))
            padded_buf = np.zeros((len(master_buffer), 2), dtype=np.float32)
            padded_buf[:usable_len] = buf[:usable_len]
            activity = detect_audio_activity(padded_buf, window_ms=50, sample_rate=SAMPLE_RATE)
            cloud_activities.append(activity)

        # Calculate overlap count at each sample position
        activity_stack = np.array(cloud_activities, dtype=np.float32)
        overlap_count = activity_stack.sum(axis=0)

        # Pan spread factor: more spread when more clouds overlap
        # 1 cloud = no extra spread, 2+ clouds = spread to opposite sides
        max_pan_spread = 0.7  # Maximum additional pan when overlapping

        # Apply panning to each cloud layer and mix into master
        for i, (name, buf, base_pan) in enumerate(cloud_layers):
            usable_len = min(len(buf), len(master_buffer))

            # Create pan envelope based on overlap
            pan_envelope = np.full(len(master_buffer), base_pan, dtype=np.float32)

            # When overlapping, spread apart
            for j, (other_name, other_buf, other_base_pan) in enumerate(cloud_layers):
                if i == j:
                    continue
                # Where both are active, increase pan separation
                both_active = cloud_activities[i] & cloud_activities[j]

                # Direction: if this cloud's base_pan < other's, go more left, else go more right
                if base_pan <= other_base_pan:
                    spread_direction = -1.0  # Go left
                else:
                    spread_direction = 1.0   # Go right

                # Smooth the activity detection for gradual panning
                # Apply a simple smoothing (rolling average)
                smooth_window = int(SAMPLE_RATE * 0.3)  # 300ms smoothing
                if smooth_window > 1:
                    kernel = np.ones(smooth_window) / smooth_window
                    smooth_activity = np.convolve(both_active.astype(np.float32), kernel, mode='same')
                else:
                    smooth_activity = both_active.astype(np.float32)

                # Apply spread
                pan_envelope += spread_direction * max_pan_spread * smooth_activity

            # Clamp pan to valid range
            pan_envelope = np.clip(pan_envelope, -1.0, 1.0)

            # Apply panning to buffer
            padded_buf = np.zeros((len(master_buffer), 2), dtype=np.float32)
            padded_buf[:usable_len] = buf[:usable_len]
            panned_buf = apply_dynamic_pan_envelope(padded_buf, pan_envelope)

            # Mix into master
            master_buffer += panned_buf

            if verbose:
                # Calculate how much panning was applied
                active_samples = cloud_activities[i].sum()
                if active_samples > 0:
                    active_pan_values = pan_envelope[cloud_activities[i]]
                    avg_pan = np.mean(active_pan_values)
                    pan_range = np.max(active_pan_values) - np.min(active_pan_values)
                    print(f"  Mixed {name} layer (pan: base={base_pan:+.1f}, avg={avg_pan:+.2f}, range={pan_range:.2f})")
                else:
                    print(f"  Mixed {name} layer (no activity)")

        combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

    # Layer in Feedback loops (overlapping, random selection) - UNIFIED NUMPY VERSION
    if verbose:
        print("\n>>> START layer: feedback")
    t_feedback_start = time.perf_counter()
    if include_feedback:
        if verbose:
            print("\nLayering Feedback samples (NumPy engine, random selection)...")

        feedback_samples = load_feedback_samples()
        if feedback_samples:
            feedback_config = LayerConfig(
                name="Feedback",
                samples=feedback_samples,
                layer_type=LayerType.INTERVAL,
                interval_seconds=feedback_interval,
                max_overlap=2,
                gain_db=-12.0,
                selection=SelectionMode.RANDOM,
            )

            feedback_np = render_layer(
                config=feedback_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_continuous_fn=render_layer_continuous_np,
                render_interval_fn=render_layer_interval_np,
            )

            usable_len = min(len(feedback_np), len(master_buffer))
            master_buffer[:usable_len] += feedback_np[:usable_len]
            combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

            if verbose:
                print(f"  Mixed Feedback layer (NumPy)")
    layer_timings['feedback'] = time.perf_counter() - t_feedback_start
    if verbose:
        print(f"<<< END layer: feedback ({layer_timings['feedback']:.2f}s)")

    # Layer in Stylo samples (like organetta but faster) - UNIFIED NUMPY VERSION
    if verbose:
        print("\n>>> START layer: stylo")
    t_stylo_start = time.perf_counter()
    if include_stylo:
        if verbose:
            print("\nLayering Stylo samples (NumPy engine)...")

        stylo_samples = load_stylo_samples()
        if stylo_samples:
            stylo_config = LayerConfig(
                name="Stylo",
                samples=stylo_samples,
                layer_type=LayerType.INTERVAL,
                interval_seconds=stylo_interval,
                gain_db=-30.0,  # Very quiet
                selection=SelectionMode.SEQUENTIAL,
            )

            stylo_np = render_layer(
                config=stylo_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_continuous_fn=render_layer_continuous_np,
                render_interval_fn=render_layer_interval_np,
            )

            usable_len = min(len(stylo_np), len(master_buffer))
            master_buffer[:usable_len] += stylo_np[:usable_len]
            combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

            if verbose:
                print(f"  Mixed Stylo layer (NumPy)")
    layer_timings['stylo'] = time.perf_counter() - t_stylo_start
    if verbose:
        print(f"<<< END layer: stylo ({layer_timings['stylo']:.2f}s)")

    # Layer in Trichords (continuous, like bass flute) - UNIFIED NUMPY VERSION
    if verbose:
        print("\n>>> START layer: trichords")
    t_trichords_start = time.perf_counter()
    if include_trichords:
        if verbose:
            print("\nLayering Trichords samples (NumPy engine, continuous)...")

        trichords_samples = load_trichords_samples()
        if trichords_samples:
            trichords_config = LayerConfig(
                name="Trichords",
                samples=trichords_samples,
                layer_type=LayerType.CONTINUOUS,
                gain_db=-6.0,
                selection=SelectionMode.SEQUENTIAL,
            )

            trichords_np = render_layer(
                config=trichords_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_continuous_fn=render_layer_continuous_np,
                render_interval_fn=render_layer_interval_np,
                render_cloud_fn=render_layer_cloud_np,
            )

            usable_len = min(len(trichords_np), len(master_buffer))
            master_buffer[:usable_len] += trichords_np[:usable_len]
            combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

            if verbose:
                print(f"  Mixed Trichords layer (NumPy)")
    layer_timings['trichords'] = time.perf_counter() - t_trichords_start
    if verbose:
        print(f"<<< END layer: trichords ({layer_timings['trichords']:.2f}s)")

    # Layer in Tremolo Oct (interval, like organetta) - UNIFIED NUMPY VERSION
    if verbose:
        print("\n>>> START layer: tremolo_oct")
    t_tremolo_oct_start = time.perf_counter()
    if include_tremolo_oct:
        if verbose:
            print("\nLayering Tremolo Oct samples (NumPy engine)...")

        tremolo_oct_samples = load_tremolo_oct_samples()
        if tremolo_oct_samples:
            tremolo_oct_config = LayerConfig(
                name="Tremolo Oct",
                samples=tremolo_oct_samples,
                layer_type=LayerType.INTERVAL,
                interval_seconds=tremolo_oct_interval,
                gain_db=-6.0,
                selection=SelectionMode.SEQUENTIAL,
            )

            tremolo_oct_np = render_layer(
                config=tremolo_oct_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_continuous_fn=render_layer_continuous_np,
                render_interval_fn=render_layer_interval_np,
                render_cloud_fn=render_layer_cloud_np,
            )

            usable_len = min(len(tremolo_oct_np), len(master_buffer))
            master_buffer[:usable_len] += tremolo_oct_np[:usable_len]
            combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

            if verbose:
                print(f"  Mixed Tremolo Oct layer (NumPy)")
    layer_timings['tremolo_oct'] = time.perf_counter() - t_tremolo_oct_start
    if verbose:
        print(f"<<< END layer: tremolo_oct ({layer_timings['tremolo_oct']:.2f}s)")

    # Layer in Avery Violin (continuous, like bass flute) - UNIFIED NUMPY VERSION
    if verbose:
        print("\n>>> START layer: averyviolin")
    t_averyviolin_start = time.perf_counter()
    if include_averyviolin:
        if verbose:
            print("\nLayering Avery Violin samples (NumPy engine, continuous)...")

        averyviolin_dict = load_averyviolin_samples()
        if averyviolin_dict:
            av_sample_list = [{"name": name, "pitch_classes": data["pitch_classes"],
                              "audio_path": data["audio_path"]}
                             for name, data in sorted(averyviolin_dict.items())]

            if av_sample_list:
                averyviolin_config = LayerConfig(
                    name="Avery Violin",
                    samples=av_sample_list,
                    layer_type=LayerType.CONTINUOUS,
                    gain_db=-2.0,  # Boosted from -6.0
                )

                averyviolin_np = render_layer(
                    config=averyviolin_config,
                    harmonic_start_times=harmonic_start_times,
                    harmonic_events=harmonic_events_sorted,
                    find_transposition_fn=find_bassflute_transposition,
                    total_duration_ms=total_duration_ms,
                    sample_rate=SAMPLE_RATE,
                    channels=CHANNELS,
                    verbose=verbose,
                    seed=seed,
                    render_continuous_fn=render_layer_continuous_np,
                    render_interval_fn=render_layer_interval_np,
                )

                usable_len = min(len(averyviolin_np), len(master_buffer))
                master_buffer[:usable_len] += averyviolin_np[:usable_len]
                combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

                if verbose:
                    print(f"  Mixed Avery Violin layer (NumPy)")
    layer_timings['averyviolin'] = time.perf_counter() - t_averyviolin_start
    if verbose:
        print(f"<<< END layer: averyviolin ({layer_timings['averyviolin']:.2f}s)")

    # Layer in Dictamel (continuous, like bass flute) - UNIFIED NUMPY VERSION
    if verbose:
        print("\n>>> START layer: dictamel")
    t_dictamel_start = time.perf_counter()
    if include_dictamel:
        if verbose:
            print("\nLayering Dictamel samples (NumPy engine, continuous)...")

        dictamel_dict = load_dictamel_samples()
        if dictamel_dict:
            dm_sample_list = [{"name": name, "pitch_classes": data["pitch_classes"],
                              "audio_path": data["audio_path"]}
                             for name, data in sorted(dictamel_dict.items())]

            if dm_sample_list:
                dictamel_config = LayerConfig(
                    name="Dictamel",
                    samples=dm_sample_list,
                    layer_type=LayerType.CONTINUOUS,
                    gain_db=-6.0,
                )

                dictamel_np = render_layer(
                    config=dictamel_config,
                    harmonic_start_times=harmonic_start_times,
                    harmonic_events=harmonic_events_sorted,
                    find_transposition_fn=find_bassflute_transposition,
                    total_duration_ms=total_duration_ms,
                    sample_rate=SAMPLE_RATE,
                    channels=CHANNELS,
                    verbose=verbose,
                    seed=seed,
                    render_continuous_fn=render_layer_continuous_np,
                    render_interval_fn=render_layer_interval_np,
                )

                usable_len = min(len(dictamel_np), len(master_buffer))
                master_buffer[:usable_len] += dictamel_np[:usable_len]
                combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

                if verbose:
                    print(f"  Mixed Dictamel layer (NumPy)")
    layer_timings['dictamel'] = time.perf_counter() - t_dictamel_start
    if verbose:
        print(f"<<< END layer: dictamel ({layer_timings['dictamel']:.2f}s)")

    # Layer in Scelsi Pezzi (interval-based, like organetta - transposes to fit harmony)
    if verbose:
        print("\n>>> START layer: scelsipezzi")
    t_scelsipezzi_start = time.perf_counter()
    if include_scelsipezzi:
        if verbose:
            print(f"\nLayering Scelsi Pezzi samples (NumPy engine, every {scelsipezzi_interval}s)...")

        scelsipezzi_dict = load_scelsipezzi_samples()
        if scelsipezzi_dict:
            sp_sample_list = [{"name": name, "pitch_classes": data["pitch_classes"],
                              "audio_path": data["audio_path"]}
                             for name, data in sorted(scelsipezzi_dict.items())]

            if sp_sample_list:
                scelsipezzi_config = LayerConfig(
                    name="Scelsi Pezzi",
                    samples=sp_sample_list,
                    layer_type=LayerType.INTERVAL,
                    interval_seconds=scelsipezzi_interval,
                    gain_db=-6.0,
                    selection=SelectionMode.SEQUENTIAL,
                )

                scelsipezzi_np = render_layer(
                    config=scelsipezzi_config,
                    harmonic_start_times=harmonic_start_times,
                    harmonic_events=harmonic_events_sorted,
                    find_transposition_fn=find_bassflute_transposition,
                    total_duration_ms=total_duration_ms,
                    sample_rate=SAMPLE_RATE,
                    channels=CHANNELS,
                    verbose=verbose,
                    seed=seed,
                    render_continuous_fn=render_layer_continuous_np,
                    render_interval_fn=render_layer_interval_np,
                )

                usable_len = min(len(scelsipezzi_np), len(master_buffer))
                master_buffer[:usable_len] += scelsipezzi_np[:usable_len]
                combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

                if verbose:
                    print(f"  Mixed Scelsi Pezzi layer (NumPy)")
    layer_timings['scelsipezzi'] = time.perf_counter() - t_scelsipezzi_start
    if verbose:
        print(f"<<< END layer: scelsipezzi ({layer_timings['scelsipezzi']:.2f}s)")

    # Layer in Godette (chord-triggered, one at a time) - CHORD-TRIGGERED VERSION
    if verbose:
        print("\n>>> START layer: godette")
    t_godette_start = time.perf_counter()
    if include_godette:
        if verbose:
            print("\nLayering Godette samples (NumPy engine, chord-triggered)...")

        godette_samples = load_godette_samples()
        if godette_samples:
            godette_config = LayerConfig(
                name="Godette",
                samples=godette_samples,
                layer_type=LayerType.CHORD_TRIGGERED,
                gain_db=0.0,
                selection=SelectionMode.SEQUENTIAL,
            )

            godette_np = render_layer(
                config=godette_config,
                harmonic_start_times=harmonic_start_times,
                harmonic_events=harmonic_events_sorted,
                find_transposition_fn=find_bassflute_transposition,
                total_duration_ms=total_duration_ms,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                verbose=verbose,
                seed=seed,
                render_chord_triggered_fn=render_layer_chord_triggered_np,
            )

            usable_len = min(len(godette_np), len(master_buffer))
            master_buffer[:usable_len] += godette_np[:usable_len]
            combined = buffer_to_audiosegment(master_buffer, SAMPLE_RATE)

            if verbose:
                print(f"  Mixed Godette layer (NumPy)")
    layer_timings['godette'] = time.perf_counter() - t_godette_start
    if verbose:
        print(f"<<< END layer: godette ({layer_timings['godette']:.2f}s)")

    # Final normalization to prevent clipping
    # Normalize to -3 dB to leave headroom while maximizing volume
    if verbose:
        print("\nApplying final normalization...")
    combined = normalize_audio_peak(combined, -3.0)
    if verbose:
        print(f"  Normalized final mix to -3 dB peak")

    # Truncate to max duration if specified
    if max_duration_ms is not None and len(combined) > max_duration_ms:
        if verbose:
            print(f"\nTruncating from {len(combined)/1000:.2f}s to {max_duration_ms/1000:.2f}s")
        combined = combined[:max_duration_ms]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="wav")

    if verbose:
        print(f"\nExported: {output_path}")
        print(f"Duration: {len(combined) / 1000:.2f}s")

    # Export MIDI files if generated
    if include_midi_chords and chord_events and 'voicing_midi_obj' in locals():
        voicing_midi_path = output_path.with_name(output_path.stem + "_voicing.mid")
        bass_midi_path = output_path.with_name(output_path.stem + "_bass.mid")

        voicing_midi_obj.write(str(voicing_midi_path))
        bass_midi_obj.write(str(bass_midi_path))

        if verbose:
            print(f"Exported: {voicing_midi_path}")
            print(f"Exported: {bass_midi_path}")

    return output_path


def load_all_samples(
    include_handel: bool = True,
    include_progressions: bool = True,
    include_glaz_sax: bool = True,
    include_hyacinthe: bool = True,
    include_kraus: bool = True
) -> Tuple[Dict[str, Dict], Dict[str, Path]]:
    """
    Load samples from all enabled libraries.

    Returns:
        samples: Combined sample data dict
        audio_dirs: Mapping from sample name to its audio directory
    """
    samples = {}
    audio_dirs = {}

    # Load Feldman samples
    with open(MANIFEST_PATH) as f:
        feldman_samples = json.load(f)

    for name, data in feldman_samples.items():
        if name.startswith("_"):
            continue
        samples[name] = data
        audio_dirs[name] = AUDIO_DIR

    # Load Handel samples if enabled
    if include_handel and HANDEL_MANIFEST_PATH.exists():
        with open(HANDEL_MANIFEST_PATH) as f:
            handel_samples = json.load(f)

        for name, data in handel_samples.items():
            if name.startswith("_"):
                continue
            # Prefix to avoid name collisions
            prefixed_name = f"handel_{name}"
            samples[prefixed_name] = data
            audio_dirs[prefixed_name] = HANDEL_AUDIO_DIR

    # Load progression samples if enabled
    # Only include samples with verified MIDI/audio sync (Perfect score)
    if include_progressions:
        # Glaz Sax Chorale - all 3 now have manual timing
        if include_glaz_sax and GLAZ_SAX_MANIFEST_PATH.exists():
            with open(GLAZ_SAX_MANIFEST_PATH) as f:
                glaz_samples = json.load(f)
            valid_glaz = ["glaz_sax_chorale-01", "glaz_sax_chorale-02", "glaz_sax_chorale-03"]
            for name in valid_glaz:
                if name in glaz_samples:
                    data = glaz_samples[name]
                    # Convert progression format to chain format
                    prefixed_name = f"glaz_{name}"
                    samples[prefixed_name] = {
                        "first_pcs": set(data["first_chord"]["pitch_classes"]),
                        "last_pcs": set(data["last_chord"]["pitch_classes"]),
                        "duration_ms": data["duration"] * 1000,  # Convert seconds to ms
                        "chord_sequence": data.get("chord_sequence", []),
                        "first_chord": data["first_chord"],
                        "last_chord": data["last_chord"],
                        "_source_name": name,  # Original filename without prefix
                    }
                    audio_dirs[prefixed_name] = GLAZ_SAX_AUDIO_DIR

        # Hyacinthe - 01-04 now have manual timing (05 excluded, too long)
        if include_hyacinthe and HYACINTHE_MANIFEST_PATH.exists():
            with open(HYACINTHE_MANIFEST_PATH) as f:
                hyacinthe_samples = json.load(f)
            valid_hyacinthe = ["hyacinthe_01", "hyacinthe_02", "hyacinthe_03", "hyacinthe_04"]
            for name in valid_hyacinthe:
                if name in hyacinthe_samples:
                    data = hyacinthe_samples[name]
                    prefixed_name = f"hyacinthe_{name}"
                    samples[prefixed_name] = {
                        "first_pcs": set(data["first_chord"]["pitch_classes"]),
                        "last_pcs": set(data["last_chord"]["pitch_classes"]),
                        "duration_ms": data["duration"] * 1000,  # Convert seconds to ms
                        "chord_sequence": data.get("chord_sequence", []),
                        "first_chord": data["first_chord"],
                        "last_chord": data["last_chord"],
                        "_source_name": name,
                    }
                    audio_dirs[prefixed_name] = HYACINTHE_AUDIO_DIR

        # Kraus Chorale - valid: 1, 2, 3, 4 (all have perfect sync)
        if include_kraus and KRAUS_MANIFEST_PATH.exists():
            with open(KRAUS_MANIFEST_PATH) as f:
                kraus_samples = json.load(f)
            valid_kraus = ["KrausChorale-1", "KrausChorale-2", "KrausChorale-3", "KrausChorale-4"]
            for name in valid_kraus:
                if name in kraus_samples:
                    data = kraus_samples[name]
                    prefixed_name = f"kraus_{name}"
                    samples[prefixed_name] = {
                        "first_pcs": set(data["first_chord"]["pitch_classes"]),
                        "last_pcs": set(data["last_chord"]["pitch_classes"]),
                        "duration_ms": data["duration"] * 1000,  # Convert seconds to ms
                        "chord_sequence": data.get("chord_sequence", []),
                        "first_chord": data["first_chord"],
                        "last_chord": data["last_chord"],
                        "_source_name": name,
                    }
                    audio_dirs[prefixed_name] = KRAUS_AUDIO_DIR

    return samples, audio_dirs


def main():
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Chain samples using Quadruple Hierarchy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-uses", type=int, default=2, help="Max uses per sample")
    parser.add_argument("--start", type=str, default=None, help="Starting sample name")
    parser.add_argument("--no-midi", action="store_true", help="Disable MIDI chord layer")
    parser.add_argument("--no-synth-bass", action="store_true", help="Disable synth bass in audio (MIDI file still generated)")
    parser.add_argument("--no-midi-synth", action="store_true", help="Disable ALL MIDI synth in audio (MIDI files still exported)")
    parser.add_argument("--no-bassflute", action="store_true", help="Disable bass flute layer")
    parser.add_argument("--brodero", action="store_true", help="Enable Brodero cloud layer (16th note bursts with silence)")
    parser.add_argument("--jicello", action="store_true", help="Enable Jicello distributed layer")
    parser.add_argument("--organetta", action="store_true", help="Enable Organetta layer (every 4 seconds)")
    parser.add_argument("--organetta-interval", type=float, default=4.0, help="Organetta interval in seconds")
    parser.add_argument("--minorchordbeat", action="store_true", help="Enable MinorChordBeat layer (eighth notes)")
    parser.add_argument("--mutebowl", action="store_true", help="Enable MuteBowl layer (eighth notes)")
    parser.add_argument("--qualitychords", action="store_true", help="Enable quality-aware chord layer (major/minor)")
    parser.add_argument("--qualitychords-density", type=float, default=0.4, help="QualityChords density 0-1 (default: 0.4 = sporadic)")
    parser.add_argument("--progressions", action="store_true", help="Include progression samples (glaz_sax, hyacinthe, kraus) in skeleton chain pool")
    parser.add_argument("--no-glaz-sax", action="store_true", help="Exclude Glaz Sax Chorale from skeleton pool")
    parser.add_argument("--no-hyacinthe", action="store_true", help="Exclude Hyacinthe from skeleton pool")
    parser.add_argument("--no-kraus", action="store_true", help="Exclude Kraus Chorale from skeleton pool")
    parser.add_argument("--no-handel", action="store_true", help="Exclude Handel strings from chain")
    parser.add_argument("--prophetfalse", action="store_true", help="Enable Prophet False synth layer")
    parser.add_argument("--prophetfalse-interval", type=float, default=3.0, help="Prophet False interval in seconds (default: 3)")
    parser.add_argument("--harmonicker", action="store_true", help="Enable Harmonicker (harmonica) layer")
    parser.add_argument("--harmonicker-interval", type=float, default=5.0, help="Harmonicker interval in seconds (default: 5)")
    parser.add_argument("--gothicharp", action="store_true", help="Enable Gothic Harp layer (16th note clouds, 5s bursts after 7-9s silence)")
    parser.add_argument("--laken", action="store_true", help="Enable Laken layer (16th note clouds, 5s bursts after 7-9s silence, -5dB)")
    parser.add_argument("--gentleharpsi", action="store_true", help="Enable Gentle Harpsichord layer (32nd note clouds, 6s bursts after 10-11s silence)")
    parser.add_argument("--feedback", action="store_true", help="Enable Feedback layer (overlapping, random selection)")
    parser.add_argument("--feedback-interval", type=float, default=4.0, help="Feedback interval in seconds (default: 4)")
    parser.add_argument("--stylo", action="store_true", help="Enable Stylo layer (like organetta but faster, every 2s)")
    parser.add_argument("--stylo-interval", type=float, default=2.0, help="Stylo interval in seconds (default: 2)")
    parser.add_argument("--trichords", action="store_true", help="Enable Trichords layer (continuous, like bass flute)")
    parser.add_argument("--tremolo-oct", action="store_true", help="Enable Tremolo Oct layer (interval, like organetta)")
    parser.add_argument("--tremolo-oct-interval", type=float, default=4.0, help="Tremolo Oct interval in seconds (default: 4)")
    parser.add_argument("--averyviolin", action="store_true", help="Enable Avery Violin layer (continuous, like bass flute)")
    parser.add_argument("--dictamel", action="store_true", help="Enable Dictamel layer (continuous, like bass flute)")
    parser.add_argument("--scelsipezzi", action="store_true", help="Enable Scelsi Pezzi layer (interval-based, transposes to fit harmony)")
    parser.add_argument("--scelsipezzi-interval", type=float, default=8.0, help="Scelsi Pezzi interval in seconds (default: 8)")
    parser.add_argument("--godette", action="store_true", help="Enable Godette layer (chord-triggered, one at a time)")
    parser.add_argument("--json-only", action="store_true", help="Output JSON only, skip audio rendering")
    parser.add_argument("--duration", type=float, default=90.0, help="Max output duration in seconds (default: 90s = 1:30)")
    parser.add_argument("--fast-preview", action="store_true", help="Use FAST_PREVIEW mode (applies render switches)")
    parser.add_argument("--skeleton-test", action="store_true", help="Skeleton timing test: only skeleton + MIDI synth (no other layers)")
    args = parser.parse_args()

    # SKELETON TEST MODE: Only skeleton + MIDI synth chords for timing verification
    if args.skeleton_test:
        print("\n[SKELETON TEST MODE - Skeleton + MIDI synth only]")
        # Disable all layers (but NOT progressions - we want progression samples in chain pool)
        args.no_bassflute = True
        args.brodero = False
        args.jicello = False
        args.organetta = False
        args.minorchordbeat = False
        args.mutebowl = False
        args.qualitychords = False
        # Note: args.progressions is NOT set to False here - it controls sample loading
        # The progression layer rendering is naturally disabled in skeleton-test mode
        args.prophetfalse = False
        args.harmonicker = False
        args.gothicharp = False
        args.laken = False
        args.gentleharpsi = False
        args.feedback = False
        args.stylo = False
        args.trichords = False
        args.tremolo_oct = False
        args.averyviolin = False
        args.dictamel = False
        args.godette = False
        # Enable MIDI synth so we can hear chord changes
        args.no_midi = False
        args.no_midi_synth = False
        args.no_synth_bass = False  # Keep bass stabs for beat reference

    # Apply render switches from module-level constants
    # These override command-line args when set, allowing quick toggling
    use_fast_preview = args.fast_preview or FAST_PREVIEW

    if use_fast_preview:
        print("\n[FAST_PREVIEW MODE - Using render switches from module constants]")
        # Apply module-level switches
        args.brodero = ENABLE_BRODERO
        args.jicello = ENABLE_JICELLO
        args.organetta = ENABLE_ORGANETTA
        args.gentleharpsi = ENABLE_GENTLEHARPSI
        args.feedback = ENABLE_FEEDBACK
        args.gothicharp = ENABLE_CLOUDS
        args.laken = ENABLE_CLOUDS
        args.no_bassflute = not ENABLE_BASSFLUTE
        args.no_midi_synth = not ENABLE_MIDI_SYNTH
        args.no_synth_bass = not ENABLE_SYNTH_BASS
        args.progressions = ENABLE_PROGRESSIONS
        # Keep duration short for preview
        if args.duration > 60:
            args.duration = 60.0
            print(f"  Duration capped at 60s for fast preview")

    # Load samples from all libraries
    samples, audio_dirs = load_all_samples(
        include_handel=not args.no_handel,
        include_progressions=args.progressions,  # Only include if --progressions flag is set
        include_glaz_sax=not args.no_glaz_sax,
        include_hyacinthe=not args.no_hyacinthe,
        include_kraus=not args.no_kraus
    )

    print("="*60)
    print("QUADRUPLE HIERARCHY CHAIN")
    print("="*60)
    print(f"Seed: {args.seed}")
    print(f"Max uses per sample: {args.max_uses}")
    feldman_count = len([k for k in samples if not k.startswith('_') and not k.startswith('handel_') and not k.startswith('glaz_') and not k.startswith('hyacinthe_') and not k.startswith('kraus_')])
    handel_count = len([k for k in samples if k.startswith('handel_')])
    glaz_count = len([k for k in samples if k.startswith('glaz_')])
    hyacinthe_count = len([k for k in samples if k.startswith('hyacinthe_')])
    kraus_count = len([k for k in samples if k.startswith('kraus_')])
    progression_count = glaz_count + hyacinthe_count + kraus_count
    total_count = len([k for k in samples if not k.startswith('_')])
    print(f"Available samples: {total_count} (Feldman: {feldman_count}, Handel: {handel_count}, Progressions: {progression_count})")
    if progression_count > 0:
        print(f"  Progressions breakdown: GlazSax: {glaz_count}, Hyacinthe: {hyacinthe_count}, Kraus: {kraus_count}")
    print(f"MIDI chords: {'OFF' if args.no_midi else 'ON'}")
    print(f"Bass flute: {'OFF' if args.no_bassflute else 'ON'}")
    print(f"Brodero: {'ON (16th note clouds)' if args.brodero else 'OFF'}")
    print(f"Jicello: {'ON' if args.jicello else 'OFF'}")
    print(f"Organetta: {'ON @ every ' + str(args.organetta_interval) + 's' if args.organetta else 'OFF'}")
    print(f"MinorChordBeat: {'ON (eighth notes @ 120 BPM)' if args.minorchordbeat else 'OFF'}")
    print(f"MuteBowl: {'ON (eighth notes @ 120 BPM)' if args.mutebowl else 'OFF'}")
    print(f"QualityChords: {'ON @ ' + str(int(args.qualitychords_density*100)) + '% density' if args.qualitychords else 'OFF'}")
    if args.progressions:
        prog_sources = []
        if not args.no_glaz_sax:
            prog_sources.append("GlazSax")
        if not args.no_hyacinthe:
            prog_sources.append("Hyacinthe")
        if not args.no_kraus:
            prog_sources.append("Kraus")
        print(f"Progressions in skeleton pool: {', '.join(prog_sources)}")
    else:
        print(f"Progressions in skeleton pool: none")
    print(f"ProphetFalse: {'ON @ every ' + str(args.prophetfalse_interval) + 's' if args.prophetfalse else 'OFF'}")
    print(f"Harmonicker: {'ON (continuous)' if args.harmonicker else 'OFF'}")
    print(f"GothicHarp: {'ON (5s clouds after 7-9s silence)' if args.gothicharp else 'OFF'}")
    print(f"GentleHarpsi: {'ON (6s clouds of 32nd notes after 10-11s silence)' if args.gentleharpsi else 'OFF'}")
    print(f"Feedback: {'ON @ every ' + str(args.feedback_interval) + 's (random, overlapping)' if args.feedback else 'OFF'}")
    print(f"Stylo: {'ON @ every ' + str(args.stylo_interval) + 's (sequential)' if args.stylo else 'OFF'}")
    print(f"AveryViolin: {'ON (continuous)' if args.averyviolin else 'OFF'}")
    print(f"Dictamel: {'ON (continuous)' if args.dictamel else 'OFF'}")
    print(f"Godette: {'ON (chord-triggered, one at a time)' if args.godette else 'OFF'}")
    print(f"Max duration: {args.duration:.1f}s ({args.duration/60:.1f} min)")

    # Build chain
    chain = build_chain(
        samples,
        audio_dirs=audio_dirs,
        start_sample=args.start,
        max_uses=args.max_uses,
        seed=args.seed,
        verbose=True
    )

    print("\n" + "="*60)
    print(f"Chain length: {len(chain)} samples")
    total_duration = sum(link.duration_ms for link in chain)
    print(f"Total duration: {total_duration/1000:.2f}s")

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.skeleton_test:
        layer_str = "SKELETON_TEST"
    else:
        layers = []
        if not args.no_midi:
            layers.append("midi")
        if not args.no_bassflute:
            layers.append("bf")
        if args.brodero:
            layers.append("bro")
        if args.jicello:
            layers.append("jic")
        layer_str = "_".join(layers) if layers else "dry"

    output_path = OUTPUT_DIR / f"feldman_s{args.seed}_{layer_str}_{timestamp}.wav"

    # Render audio (unless json-only)
    if not args.json_only:
        render_chain_to_wav(
            chain,
            output_path,
            seed=args.seed,
            include_midi_chords=not args.no_midi,
            include_bassflute=not args.no_bassflute,
            include_brodero=args.brodero,
            include_jicello=args.jicello,
            include_organetta=args.organetta,
            organetta_interval=args.organetta_interval,
            include_minorchordbeat=args.minorchordbeat,
            include_mutebowl=args.mutebowl,
            include_qualitychords=args.qualitychords,
            qualitychords_density=args.qualitychords_density,
            # Progressions (glaz_sax, hyacinthe, kraus) are skeleton samples in the chain pool.
            # They are concatenated sequentially, not rendered as an overlay layer.
            include_prophetfalse=args.prophetfalse,
            prophetfalse_interval=args.prophetfalse_interval,
            include_harmonicker=args.harmonicker,
            harmonicker_interval=args.harmonicker_interval,
            include_gothicharp=args.gothicharp,
            include_laken=args.laken,
            include_gentleharpsi=args.gentleharpsi,
            include_feedback=args.feedback,
            feedback_interval=args.feedback_interval,
            include_stylo=args.stylo,
            stylo_interval=args.stylo_interval,
            include_trichords=args.trichords,
            include_tremolo_oct=args.tremolo_oct,
            tremolo_oct_interval=args.tremolo_oct_interval,
            include_averyviolin=args.averyviolin,
            include_dictamel=args.dictamel,
            include_scelsipezzi=args.scelsipezzi,
            scelsipezzi_interval=args.scelsipezzi_interval,
            include_godette=args.godette,
            include_synth_bass=not args.no_synth_bass,
            render_midi_synth=not args.no_midi_synth,
            max_duration_ms=int(args.duration * 1000)
        )
    else:
        print("\n[JSON-only mode - skipping audio render]")

    # Save chain data as JSON
    # Build chain data with explicit quadruple hierarchy relationships
    chain_json = []
    for i, link in enumerate(chain):
        entry = {
            "sample": link.sample,
            "transposition": link.transposition,
            "first_pcs": sorted(link.first_pcs),
            "last_pcs": sorted(link.last_pcs),
            "duration_ms": link.duration_ms,
            "chord_b_onset_ratio": link.chord_b_onset_ratio,
            # The bridging chord (quadruple hierarchy)
            "bridging_chord": {
                "name": link.inferred_chord["name"],
                "pitch_classes": link.inferred_chord["pitch_classes"],
                "this_sample_last_pcs_subset": sorted(link.last_pcs),
            },
            # For progression samples: full chord sequence with timing
            "chord_sequence": link.chord_sequence if link.chord_sequence else None
        }
        # Add next sample's first_pcs to show the subset relationship
        if i < len(chain) - 1:
            next_link = chain[i + 1]
            entry["bridging_chord"]["next_sample_first_pcs_subset"] = sorted(next_link.first_pcs)
            entry["bridging_chord"]["next_sample"] = next_link.sample
            entry["bridging_chord"]["next_transposition"] = next_link.transposition

        chain_json.append(entry)

    chain_data = {
        "seed": args.seed,
        "max_uses": args.max_uses,
        "layers": {
            "midi_chords": not args.no_midi,
            "synth_bass": not args.no_synth_bass,
            "bass_flute": not args.no_bassflute,
            "brodero": args.brodero,
            "jicello": args.jicello,
            "organetta": {"enabled": args.organetta, "interval_seconds": args.organetta_interval} if args.organetta else False,
            "minorchordbeat": args.minorchordbeat,
            "mutebowl": args.mutebowl,
            "qualitychords": {"enabled": args.qualitychords, "density": args.qualitychords_density} if args.qualitychords else False,
            # Progressions are skeleton samples, not overlay layers. They participate in chain building.
            "progressions_in_skeleton_pool": args.progressions,
            "prophetfalse": {"enabled": args.prophetfalse, "interval_seconds": args.prophetfalse_interval} if args.prophetfalse else False,
            "harmonicker": {"enabled": args.harmonicker, "interval_seconds": args.harmonicker_interval} if args.harmonicker else False,
            "gothicharp": {"enabled": args.gothicharp, "silence_range": "7-9s", "cloud_duration": "5s", "note_value": "16th"} if args.gothicharp else False,
            "laken": {"enabled": args.laken, "silence_range": "7-9s", "cloud_duration": "5s", "note_value": "16th", "volume_db": -15} if args.laken else False,
            "gentleharpsi": {"enabled": args.gentleharpsi, "silence_range": "10-11s", "cloud_duration": "6s", "note_value": "32nd"} if args.gentleharpsi else False,
            "feedback": {"enabled": args.feedback, "interval_seconds": args.feedback_interval} if args.feedback else False,
            "scelsipezzi": {"enabled": args.scelsipezzi, "interval_seconds": args.scelsipezzi_interval} if args.scelsipezzi else False,
        },
        "glissando": {
            "duration_ms": GLISSANDO_MS,
            "anticipation_ms": GLISSANDO_ANTICIPATION_MS,
        },
        "progression_sources": {
            "glaz_sax": not args.no_glaz_sax,
            "hyacinthe": not args.no_hyacinthe,
            "kraus": not args.no_kraus,
            "handel": not args.no_handel,
        },
        "chain": chain_json
    }

    # Use same base name as WAV file
    json_path = output_path.with_suffix(".json")
    with open(json_path, 'w') as f:
        json.dump(chain_data, f, indent=2)

    print(f"Chain data: {json_path}")
    print("="*60)


if __name__ == "__main__":
    main()
