#!/usr/bin/env python3
"""
Minimal skeleton-only render with MIDI export.

Purpose:
- Fast iteration for timing validation
- Single source of truth: audio render creates the clock, MIDI follows
- No overlay layers, no reactive rendering, no clouds

Outputs:
- skeleton WAV (chain audio only)
- chord MIDI (harmonic timeline)
- JSON metadata

This is the canonical timing-debugging and architecture-validation render path.
"""

import json
import random
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime

import numpy as np
import pretty_midi
from pydub import AudioSegment

# === PATHS ===
SAMPLES_DIR = Path("/Users/soney/Music/samples/3voices-feldman")
AUDIO_DIR = SAMPLES_DIR / "samples"
MANIFEST_PATH = SAMPLES_DIR / "samples_data.json"

HANDEL_DIR = Path("/Users/soney/Music/samples/handel-strings")
HANDEL_AUDIO_DIR = HANDEL_DIR / "samples"
HANDEL_MANIFEST_PATH = HANDEL_DIR / "samples_data.json"

CHORDS_PATH = Path(__file__).parent.parent / "data" / "chords_no_supersets.json"
OUTPUT_DIR = Path("/Users/soney/Github/arranger/output")

# === CONSTANTS ===
SAMPLE_RATE = 44100
CHANNELS = 2
SAMPLE_NORMALIZE_DB = -5.0


@dataclass
class ChainLink:
    """A single link in the chain."""
    sample: str
    transposition: int
    first_pcs: Set[int]
    last_pcs: Set[int]
    duration_ms: int
    chord_b_onset_ratio: float
    inferred_chord: Dict
    audio_dir: Optional[Path] = None
    chord_sequence: Optional[List[Dict]] = None


@dataclass
class RenderedSkeletonEvent:
    """Tracks actual rendered timing for a skeleton sample."""
    sample_name: str
    transposition: int
    source_duration_ms: float
    rendered_start_ms: float
    rendered_duration_ms: float
    rendered_end_ms: float
    first_pcs: Set[int]
    last_pcs: Set[int]
    inferred_chord: Dict
    chord_b_onset_ratio: float
    chord_a_start_ms: float
    chord_b_start_ms: float


@dataclass
class HarmonicEvent:
    """A chord event in the harmonic timeline."""
    start_ms: float
    end_ms: float
    chord_pcs: Set[int]
    chord_root: int
    chord_name: str


# === AUDIO CACHE ===
AUDIO_CACHE: Dict[str, Dict] = {}


def get_cached_audio(audio_path: Path, target_sample_rate: int = SAMPLE_RATE) -> Optional[Dict]:
    """Load and cache audio, resampling to target rate."""
    cache_key = f"{audio_path}:{target_sample_rate}"

    if cache_key in AUDIO_CACHE:
        return AUDIO_CACHE[cache_key]

    if not audio_path.exists():
        return None

    try:
        audio = AudioSegment.from_file(audio_path)
        if audio.frame_rate != target_sample_rate:
            audio = audio.set_frame_rate(target_sample_rate)

        raw_samples = audio.get_array_of_samples()
        samples = np.array(raw_samples, dtype=np.float32)

        # Normalize to [-1, 1]
        max_val = 32768.0 if audio.sample_width == 2 else 128.0
        samples = samples / max_val

        channels = audio.channels
        if channels == 2:
            samples = samples.reshape((-1, 2))
        else:
            samples = samples.reshape((-1, 1))

        AUDIO_CACHE[cache_key] = {
            "samples": samples,
            "sample_rate": target_sample_rate,
            "channels": channels,
            "sample_width": audio.sample_width
        }
        return AUDIO_CACHE[cache_key]
    except Exception as e:
        print(f"  WARNING: Failed to load {audio_path}: {e}")
        return None


def apply_varispeed_np(samples: np.ndarray, semitones: float) -> np.ndarray:
    """Apply constant varispeed transposition via resampling."""
    if semitones == 0:
        return samples

    rate = 2 ** (semitones / 12.0)
    original_len = len(samples)
    new_len = int(original_len / rate)

    if new_len < 1:
        return samples[:1]

    old_indices = np.linspace(0, original_len - 1, new_len)
    old_indices_floor = np.floor(old_indices).astype(int)
    old_indices_ceil = np.minimum(old_indices_floor + 1, original_len - 1)
    fracs = (old_indices - old_indices_floor).reshape(-1, 1)

    return samples[old_indices_floor] * (1 - fracs) + samples[old_indices_ceil] * fracs


def normalize_samples_peak(samples: np.ndarray, target_db: float = -5.0) -> np.ndarray:
    """Peak normalize samples to target dB."""
    peak = np.max(np.abs(samples))
    if peak < 1e-10:
        return samples

    target_linear = 10 ** (target_db / 20.0)
    return samples * (target_linear / peak)


def load_chord_dictionary() -> Dict:
    """Load chord dictionary for inference."""
    with open(CHORDS_PATH) as f:
        return json.load(f)


def infer_chord_supersets(
    pitch_classes: Set[int],
    max_results: int = 5,
    target_sizes: List[int] = [4, 5, 6],
    prev_root: Optional[int] = None
) -> List[Dict]:
    """Find chords that contain the given pitch classes."""
    chord_dict = load_chord_dictionary()
    candidates = []

    for chord_name, chord_data in chord_dict.items():
        chord_pcs = set(chord_data.get("pitch_classes", []))
        if pitch_classes.issubset(chord_pcs) and len(chord_pcs) in target_sizes:
            candidates.append({
                "name": chord_name,
                "pitch_classes": list(chord_pcs),
                "root": chord_data.get("root", 0)
            })

    # Sort by size (prefer smaller chords) and root continuity
    def score(c):
        size_score = len(c["pitch_classes"])
        root_score = 0 if prev_root is None or c["root"] == prev_root else 1
        return (size_score, root_score)

    candidates.sort(key=score)
    return candidates[:max_results]


def transpose_set(pcs: Set[int], semitones: int) -> Set[int]:
    """Transpose pitch class set."""
    return {(p + semitones) % 12 for p in pcs}


def load_samples(include_handel: bool = True) -> Tuple[Dict, Dict]:
    """Load sample manifests."""
    samples = {}
    audio_dirs = {}

    # Feldman samples
    with open(MANIFEST_PATH) as f:
        feldman = json.load(f)
    for name, data in feldman.items():
        # Skip metadata keys
        if name.startswith('_') or not isinstance(data, dict):
            continue
        samples[name] = data
        audio_dirs[name] = AUDIO_DIR

    # Handel samples (optional)
    if include_handel and HANDEL_MANIFEST_PATH.exists():
        with open(HANDEL_MANIFEST_PATH) as f:
            handel = json.load(f)
        for name, data in handel.items():
            # Skip metadata keys
            if name.startswith('_') or not isinstance(data, dict):
                continue
            prefixed = f"handel_{name}"
            samples[prefixed] = data
            audio_dirs[prefixed] = HANDEL_AUDIO_DIR

    return samples, audio_dirs


def build_chain(
    samples: Dict,
    audio_dirs: Dict,
    seed: int,
    max_uses: int = 2,
    verbose: bool = True
) -> List[ChainLink]:
    """Build the sample chain using quadruple hierarchy logic."""
    rng = random.Random(seed)

    # Track usage
    usage_count = {name: 0 for name in samples}

    # Filter to usable samples
    usable = [name for name in samples if usage_count[name] < max_uses]
    if not usable:
        return []

    chain = []

    # Start with random sample
    current_name = rng.choice(usable)
    current_data = samples[current_name]

    while True:
        # Get pitch class data
        first_pcs = set(current_data.get("first_chord", {}).get("pitch_classes", []))
        last_pcs = set(current_data.get("last_chord", {}).get("pitch_classes", []))
        duration_ms = current_data.get("duration_ms", 5000)
        chord_b_onset_ratio = current_data.get("chord_b_onset_ratio", 0.5)

        # Infer bridging chord from last_pcs
        chord_candidates = infer_chord_supersets(last_pcs, max_results=5)
        if chord_candidates:
            inferred_chord = chord_candidates[0]
        else:
            inferred_chord = {
                "name": "unknown",
                "pitch_classes": list(last_pcs),
                "root": min(last_pcs) if last_pcs else 0
            }

        # Create chain link
        link = ChainLink(
            sample=current_name,
            transposition=0,
            first_pcs=first_pcs,
            last_pcs=last_pcs,
            duration_ms=duration_ms,
            chord_b_onset_ratio=chord_b_onset_ratio,
            inferred_chord=inferred_chord,
            audio_dir=audio_dirs.get(current_name)
        )
        chain.append(link)
        usage_count[current_name] += 1

        if verbose:
            print(f"{len(chain)}. {current_name}")

        # Find next sample
        bridging_pcs = set(inferred_chord["pitch_classes"])
        candidates = []

        for name, data in samples.items():
            if usage_count[name] >= max_uses:
                continue

            next_first = set(data.get("first_chord", {}).get("pitch_classes", []))

            # Check if first_pcs subset of bridging chord (with transposition)
            for trans in range(-6, 7):
                transposed = transpose_set(next_first, trans)
                if transposed.issubset(bridging_pcs):
                    candidates.append((name, trans))
                    break

        if not candidates:
            break

        # Pick next sample
        next_name, next_trans = rng.choice(candidates)
        next_data = samples[next_name]

        # Apply transposition
        current_name = next_name
        current_data = next_data
        chain[-1] = chain[-1]  # No change needed, trans applied to next

        # Actually we need to track trans for next sample
        if chain:
            # Find next and apply its transposition
            pass

        # Rebuild for next iteration
        first_pcs = set(next_data.get("first_chord", {}).get("pitch_classes", []))
        last_pcs = set(next_data.get("last_chord", {}).get("pitch_classes", []))

        # Transpose
        first_pcs = transpose_set(first_pcs, next_trans)
        last_pcs = transpose_set(last_pcs, next_trans)

        duration_ms = next_data.get("duration_ms", 5000)
        chord_b_onset_ratio = next_data.get("chord_b_onset_ratio", 0.5)

        # Infer chord from transposed last_pcs
        chord_candidates = infer_chord_supersets(last_pcs, max_results=5)
        if chord_candidates:
            inferred_chord = chord_candidates[0]
        else:
            inferred_chord = {
                "name": "unknown",
                "pitch_classes": list(last_pcs),
                "root": min(last_pcs) if last_pcs else 0
            }

        link = ChainLink(
            sample=next_name,
            transposition=next_trans,
            first_pcs=first_pcs,
            last_pcs=last_pcs,
            duration_ms=duration_ms,
            chord_b_onset_ratio=chord_b_onset_ratio,
            inferred_chord=inferred_chord,
            audio_dir=audio_dirs.get(next_name)
        )
        chain.append(link)
        usage_count[next_name] += 1

        if verbose:
            trans_str = f" (trans {next_trans:+d})" if next_trans else ""
            print(f"{len(chain)}. {next_name}{trans_str}")

        current_name = next_name
        current_data = next_data

    return chain


def build_harmonic_timeline_from_rendered(
    rendered_events: List[RenderedSkeletonEvent]
) -> List[HarmonicEvent]:
    """Build harmonic timeline from rendered timing."""
    timeline = []

    for i, re in enumerate(rendered_events):
        # Check if 1-chord sample
        is_one_chord = re.first_pcs == re.last_pcs

        # CHORD A: continuation from previous or infer from first_pcs
        if timeline:
            chord_a_pcs = timeline[-1].chord_pcs
            chord_a_root = timeline[-1].chord_root
            chord_a_name = timeline[-1].chord_name + "_cont"
        else:
            candidates = infer_chord_supersets(re.first_pcs, max_results=1)
            if candidates:
                chord_a_pcs = set(candidates[0]["pitch_classes"])
                chord_a_root = candidates[0]["root"]
                chord_a_name = candidates[0]["name"]
            else:
                chord_a_pcs = re.first_pcs
                chord_a_root = min(re.first_pcs) if re.first_pcs else 0
                chord_a_name = "first_chord"

        # Chord A timing
        chord_a_start_ms = re.chord_a_start_ms
        chord_a_end_ms = re.rendered_end_ms + 5000 if is_one_chord else re.chord_b_start_ms

        timeline.append(HarmonicEvent(
            start_ms=chord_a_start_ms,
            end_ms=chord_a_end_ms,
            chord_pcs=chord_a_pcs,
            chord_root=chord_a_root,
            chord_name=chord_a_name,
        ))

        # CHORD B: from inferred_chord (at chord_b_start_ms)
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


def render_skeleton(
    chain: List[ChainLink],
    max_duration_ms: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, List[RenderedSkeletonEvent], float]:
    """
    Render skeleton chain to numpy buffer.
    Returns: (audio_buffer, rendered_events, total_duration_ms)
    """
    chain_samples_list = []
    rendered_events = []
    rendered_cursor_ms = 0.0

    for i, link in enumerate(chain):
        # Get audio path
        sample_audio_dir = link.audio_dir if link.audio_dir else AUDIO_DIR
        sample_filename = link.sample

        # Strip prefixes
        for prefix in ["handel_", "glaz_", "hyacinthe_", "kraus_"]:
            if sample_filename.startswith(prefix):
                sample_filename = sample_filename[len(prefix):]
                break

        audio_path = sample_audio_dir / f"{sample_filename}.wav"

        # Load audio
        cached = get_cached_audio(audio_path, SAMPLE_RATE)
        if cached is None:
            print(f"  WARNING: {audio_path} not found, skipping")
            continue

        samples = cached["samples"].copy()

        # Apply transposition
        if link.transposition != 0:
            samples = apply_varispeed_np(samples, float(link.transposition))

        # Calculate rendered timing
        rendered_duration_ms = len(samples) * 1000.0 / SAMPLE_RATE
        rendered_start_ms = rendered_cursor_ms
        rendered_end_ms = rendered_start_ms + rendered_duration_ms

        chord_a_start_ms = rendered_start_ms
        chord_b_start_ms = rendered_start_ms + (rendered_duration_ms * link.chord_b_onset_ratio)

        # Create rendered event
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

        # Normalize
        samples = normalize_samples_peak(samples, SAMPLE_NORMALIZE_DB)

        # Ensure stereo
        if samples.ndim == 1:
            samples = np.stack([samples, samples], axis=1)
        elif samples.shape[1] == 1:
            samples = np.hstack([samples, samples])
        elif samples.shape[1] > 2:
            samples = samples[:, :2]

        chain_samples_list.append((samples, len(samples)))
        rendered_cursor_ms = rendered_end_ms

        if verbose:
            trans_str = f" (trans {link.transposition:+d})" if link.transposition else ""
            print(f"  {link.sample}{trans_str}: {int(rendered_duration_ms)}ms")

        # Check duration limit
        if max_duration_ms and rendered_cursor_ms >= max_duration_ms:
            break

    # Build master buffer
    total_samples = sum(n for _, n in chain_samples_list)
    master_buffer = np.zeros((total_samples, CHANNELS), dtype=np.float32)

    current_sample = 0
    for samples, num_samples in chain_samples_list:
        master_buffer[current_sample:current_sample + num_samples] = samples
        current_sample += num_samples

    # Truncate if needed
    if max_duration_ms:
        max_samples = int(max_duration_ms * SAMPLE_RATE / 1000)
        if len(master_buffer) > max_samples:
            master_buffer = master_buffer[:max_samples]
            rendered_cursor_ms = max_duration_ms

    return master_buffer, rendered_events, rendered_cursor_ms


def generate_chord_midi(
    harmonic_timeline: List[HarmonicEvent],
    total_duration_ms: float
) -> pretty_midi.PrettyMIDI:
    """Generate MIDI from harmonic timeline."""
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name="Chords")

    for i, he in enumerate(harmonic_timeline):
        start_sec = he.start_ms / 1000.0

        # Duration to next event or end
        if i < len(harmonic_timeline) - 1:
            end_sec = harmonic_timeline[i + 1].start_ms / 1000.0
        else:
            end_sec = total_duration_ms / 1000.0

        # Emit chord root as single note
        root_midi = 48 + he.chord_root  # C3 + root
        note = pretty_midi.Note(
            velocity=80,
            pitch=root_midi,
            start=start_sec,
            end=end_sec
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    return midi


def export_wav(buffer: np.ndarray, path: Path, sample_rate: int = SAMPLE_RATE):
    """Export numpy buffer to WAV."""
    # Normalize to prevent clipping
    peak = np.max(np.abs(buffer))
    if peak > 0:
        target = 10 ** (-3.0 / 20.0)  # -3 dB
        buffer = buffer * (target / peak)

    # Convert to int16
    buffer_int = (buffer * 32767).astype(np.int16)

    # Create AudioSegment
    audio = AudioSegment(
        buffer_int.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=buffer.shape[1] if buffer.ndim > 1 else 1
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(path, format="wav")


def main():
    parser = argparse.ArgumentParser(description="Render skeleton chain with MIDI")
    parser.add_argument("--seed", type=int, default=789, help="Random seed")
    parser.add_argument("--duration", type=float, default=30.0, help="Max duration in seconds")
    parser.add_argument("--handel", action="store_true", default=True, help="Include Handel strings")
    parser.add_argument("--no-handel", action="store_true", help="Exclude Handel strings")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    args = parser.parse_args()

    include_handel = args.handel and not args.no_handel
    output_dir = args.output or OUTPUT_DIR
    max_duration_ms = int(args.duration * 1000)

    print("="*60)
    print("SKELETON RENDER WITH MIDI")
    print("="*60)
    print(f"Seed: {args.seed}")
    print(f"Duration: {args.duration}s")
    print(f"Handel: {'ON' if include_handel else 'OFF'}")
    print()

    timings = {}

    # === LOAD SAMPLES ===
    t0 = time.perf_counter()
    samples, audio_dirs = load_samples(include_handel=include_handel)
    timings['load_samples'] = time.perf_counter() - t0
    print(f"Loaded {len(samples)} samples in {timings['load_samples']:.2f}s")

    # === BUILD CHAIN ===
    print("\nBuilding chain...")
    t0 = time.perf_counter()
    chain = build_chain(samples, audio_dirs, seed=args.seed, verbose=args.verbose)
    timings['build_chain'] = time.perf_counter() - t0
    print(f"\nChain: {len(chain)} samples in {timings['build_chain']:.2f}s")

    # === RENDER SKELETON ===
    print("\nRendering skeleton...")
    t0 = time.perf_counter()
    audio_buffer, rendered_events, total_duration_ms = render_skeleton(
        chain, max_duration_ms=max_duration_ms, verbose=args.verbose
    )
    timings['render_skeleton'] = time.perf_counter() - t0
    print(f"\nSkeleton: {total_duration_ms/1000:.2f}s in {timings['render_skeleton']:.2f}s")

    # === TIMING AUDIT ===
    print("\n" + "="*60)
    print("TIMING AUDIT (first 10 samples)")
    print("="*60)
    print(f"{'Sample':<25} {'Trans':>5} {'Start':>10} {'Dur':>10} {'End':>10} {'ChB_Ratio':>10} {'ChordB':>10}")
    print("-"*90)

    for i, re in enumerate(rendered_events[:10]):
        print(f"{re.sample_name:<25} {re.transposition:>+5} {re.rendered_start_ms:>10.0f} {re.rendered_duration_ms:>10.0f} {re.rendered_end_ms:>10.0f} {re.chord_b_onset_ratio:>10.3f} {re.chord_b_start_ms:>10.0f}")

        # Assertion 1: start == prev end
        if i > 0:
            prev_end = rendered_events[i-1].rendered_end_ms
            if abs(re.rendered_start_ms - prev_end) > 0.1:
                print(f"  ⚠️  GAP: start={re.rendered_start_ms:.1f} != prev_end={prev_end:.1f}")

        # Assertion 2: chord_b in bounds
        if not (re.rendered_start_ms <= re.chord_b_start_ms <= re.rendered_end_ms):
            print(f"  ⚠️  CHORD_B OUT OF BOUNDS")

    # === BUILD HARMONIC TIMELINE ===
    print("\nBuilding harmonic timeline...")
    t0 = time.perf_counter()
    harmonic_timeline = build_harmonic_timeline_from_rendered(rendered_events)
    timings['build_timeline'] = time.perf_counter() - t0
    print(f"Timeline: {len(harmonic_timeline)} events in {timings['build_timeline']:.4f}s")

    # === GENERATE MIDI ===
    print("\nGenerating MIDI...")
    t0 = time.perf_counter()
    midi = generate_chord_midi(harmonic_timeline, total_duration_ms)
    timings['generate_midi'] = time.perf_counter() - t0

    # === EXPORT ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"skeleton_s{args.seed}_{timestamp}"

    wav_path = output_dir / f"{base_name}.wav"
    mid_path = output_dir / f"{base_name}.mid"
    json_path = output_dir / f"{base_name}.json"

    print("\nExporting...")
    t0 = time.perf_counter()
    export_wav(audio_buffer, wav_path)
    timings['export_wav'] = time.perf_counter() - t0
    print(f"  WAV: {wav_path} ({timings['export_wav']:.2f}s)")

    t0 = time.perf_counter()
    midi.write(str(mid_path))
    timings['export_midi'] = time.perf_counter() - t0
    print(f"  MIDI: {mid_path} ({timings['export_midi']:.4f}s)")

    # Export JSON metadata
    t0 = time.perf_counter()
    metadata = {
        "seed": args.seed,
        "duration_ms": total_duration_ms,
        "sample_rate": SAMPLE_RATE,
        "include_handel": include_handel,
        "chain_length": len(chain),
        "harmonic_events": len(harmonic_timeline),
        "rendered_events": [
            {
                "sample_name": re.sample_name,
                "transposition": re.transposition,
                "rendered_start_ms": re.rendered_start_ms,
                "rendered_duration_ms": re.rendered_duration_ms,
                "rendered_end_ms": re.rendered_end_ms,
                "chord_b_start_ms": re.chord_b_start_ms,
            }
            for re in rendered_events
        ],
        "harmonic_timeline": [
            {
                "start_ms": he.start_ms,
                "end_ms": he.end_ms,
                "chord_name": he.chord_name,
                "chord_root": he.chord_root,
                "chord_pcs": list(he.chord_pcs),
            }
            for he in harmonic_timeline
        ]
    }
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    timings['export_json'] = time.perf_counter() - t0
    print(f"  JSON: {json_path} ({timings['export_json']:.4f}s)")

    # === TIMING SUMMARY ===
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    total = sum(timings.values())
    for name, t in timings.items():
        pct = (t / total * 100) if total > 0 else 0
        print(f"  {name:<20}: {t:>8.3f}s ({pct:>5.1f}%)")
    print(f"  {'TOTAL':<20}: {total:>8.3f}s")
    print("="*60)


if __name__ == "__main__":
    main()
