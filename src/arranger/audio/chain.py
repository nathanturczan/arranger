"""Audio concatenation via scale-based navigation.

This module chains audio samples by walking through the Tymoczko 57-scale
network, selecting samples that fit the current scale or adjacent scales.
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydub import AudioSegment

from ..scales.tymoczko import (
    load_scales_data,
    get_adjacent_scales,
    transpose_pitch_classes,
    pitch_class_to_note_name,
)


def load_samples_manifest(project_path: Path) -> Dict:
    """Load samples_data.json from a project directory."""
    manifest_path = project_path / "samples_data.json"
    if not manifest_path.exists():
        return {}

    with open(manifest_path) as f:
        return json.load(f)


def get_original_sample_name(sample_filename: str) -> str:
    """Extract the original sample name (without transposition suffix).

    Example: 'wildrose_1_transposed_by-3.wav' -> 'wildrose_1'
    """
    name = Path(sample_filename).stem
    if "_transposed_by" in name:
        name = name.split("_transposed_by")[0]
    return name


def find_initial_samples(
    scales_dir: Path,
    sample_prefix: str,
    measure_number: int = 1,
) -> List[Tuple[str, Path]]:
    """Find all samples matching a pattern across all scale folders.

    Handles both naming conventions:
    - wildrose style: wildrose_1_, wildrose_2_
    - sarabande style: satie_sarabande_0001, satie_sarabande_0002
    """
    # Try both patterns
    patterns = [
        f"{sample_prefix}_{measure_number}_",           # wildrose_1_
        f"{sample_prefix}_{measure_number:04d}",        # satie_sarabande_0001
        f"{sample_prefix}_{measure_number:02d}",        # prefix_01
    ]

    results = []

    for scale_folder in scales_dir.iterdir():
        if not scale_folder.is_dir():
            continue

        for sample_file in scale_folder.iterdir():
            for pattern in patterns:
                if sample_file.name.startswith(pattern):
                    results.append((scale_folder.name, sample_file))
                    break

    return results


def find_next_samples(
    scales_dir: Path,
    scales_data: Dict,
    current_scale: str,
    sample_prefix: str,
    measure_number: int,
) -> List[Path]:
    """Find samples for the next measure from current and adjacent scales.

    Handles both naming conventions:
    - wildrose style: wildrose_1_, wildrose_2_
    - sarabande style: satie_sarabande_0001, satie_sarabande_0002
    """
    # Try both patterns
    patterns = [
        f"{sample_prefix}_{measure_number}_",           # wildrose_1_
        f"{sample_prefix}_{measure_number:04d}",        # satie_sarabande_0001
        f"{sample_prefix}_{measure_number:02d}",        # prefix_01
    ]

    adjacent = get_adjacent_scales(scales_data, current_scale)
    search_dirs = [current_scale] + adjacent

    samples = []
    seen = set()

    for scale_name in search_dirs:
        scale_dir = scales_dir / scale_name
        if not scale_dir.exists():
            continue

        for sample_file in scale_dir.iterdir():
            for pattern in patterns:
                if sample_file.name.startswith(pattern):
                    resolved = sample_file.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        samples.append(sample_file)
                    break

    return samples


def extract_transposition(sample_path: Path) -> int:
    """Extract transposition amount from sample filename."""
    name = sample_path.stem
    if "_transposed_by" in name:
        try:
            trans_str = name.split("_transposed_by")[-1]
            trans_str = trans_str.split("_")[0].split(".")[0]
            return int(trans_str)
        except ValueError:
            pass
    return 0


def concatenate_audio(
    scales_dir: Path,
    sample_prefix: str,
    num_measures: int,
    output_path: Optional[Path] = None,
    scales_data_path: Optional[Path] = None,
    crossfade_ms: int = 1,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Path, List[Dict]]:
    """Concatenate audio samples via scale network navigation.

    Returns:
        Tuple of (output_path, journey) where journey includes:
        - measure: measure number
        - sample: sample filename
        - scale: scale name
        - transposition: semitones transposed
        - duration_ms: duration in milliseconds
        - start_ms: start time in milliseconds
        - original_pitch_classes: original sample pitch classes
        - transposed_pitch_classes: pitch classes after transposition
        - scale_pitch_classes: pitch classes of the scale
    """
    if seed is not None:
        random.seed(seed)

    scales_data = load_scales_data(scales_data_path)

    # Load samples manifest for pitch class info
    project_path = scales_dir.parent
    samples_manifest = load_samples_manifest(project_path)

    # Find starting options
    initial_options = find_initial_samples(scales_dir, sample_prefix, 1)

    if not initial_options:
        raise ValueError(f"No samples found matching '{sample_prefix}_1_' in {scales_dir}")

    # Pick random starting point
    current_scale, current_sample = random.choice(initial_options)

    # Initialize audio
    combined = AudioSegment.from_file(current_sample)
    current_time_ms = 0

    # Get pitch class info for first sample
    transposition = extract_transposition(current_sample)
    original_name = get_original_sample_name(current_sample.name)
    original_pcs = samples_manifest.get(original_name, {}).get("pitch_classes", [])

    # If pitch_classes not in manifest, try to compute from note_names
    if not original_pcs:
        note_names = samples_manifest.get(original_name, {}).get("note_names", [])
        if note_names:
            from ..scales.tymoczko import note_name_to_pitch_class
            original_pcs = [note_name_to_pitch_class(n) for n in note_names]

    transposed_pcs = transpose_pitch_classes(original_pcs, transposition) if original_pcs else []
    scale_pcs = scales_data.get(current_scale, {}).get("pitch_classes", [])

    duration_ms = len(combined)

    # Track the journey with full pitch class info
    journey = [{
        "measure": 1,
        "sample": current_sample.name,
        "scale": current_scale,
        "transposition": transposition,
        "duration_ms": duration_ms,
        "start_ms": current_time_ms,
        "original_pitch_classes": original_pcs,
        "transposed_pitch_classes": transposed_pcs,
        "scale_pitch_classes": scale_pcs,
    }]

    current_time_ms += duration_ms

    if verbose:
        print(f"1: {current_scale} - {current_sample.name}")

    # Chain subsequent measures
    for measure in range(2, num_measures + 1):
        options = find_next_samples(
            scales_dir, scales_data, current_scale, sample_prefix, measure
        )

        if not options:
            if verbose:
                print(f"No samples found for measure {measure}, stopping")
            break

        # Pick random next sample
        next_sample = random.choice(options)
        current_scale = next_sample.parent.name

        # Get audio and timing
        next_audio = AudioSegment.from_file(next_sample)
        duration_ms = len(next_audio)

        # Get pitch class info
        transposition = extract_transposition(next_sample)
        original_name = get_original_sample_name(next_sample.name)
        original_pcs = samples_manifest.get(original_name, {}).get("pitch_classes", [])

        if not original_pcs:
            note_names = samples_manifest.get(original_name, {}).get("note_names", [])
            if note_names:
                from ..scales.tymoczko import note_name_to_pitch_class
                original_pcs = [note_name_to_pitch_class(n) for n in note_names]

        transposed_pcs = transpose_pitch_classes(original_pcs, transposition) if original_pcs else []
        scale_pcs = scales_data.get(current_scale, {}).get("pitch_classes", [])

        combined = combined.append(next_audio, crossfade=crossfade_ms)

        journey.append({
            "measure": measure,
            "sample": next_sample.name,
            "scale": current_scale,
            "transposition": transposition,
            "duration_ms": duration_ms,
            "start_ms": current_time_ms,
            "original_pitch_classes": original_pcs,
            "transposed_pitch_classes": transposed_pcs,
            "scale_pitch_classes": scale_pcs,
        })

        current_time_ms += duration_ms

        if verbose:
            print(f"{measure}: {current_scale} - {next_sample.name}")

    # Export audio
    if output_path is None:
        output_path = scales_dir.parent / "output" / f"{sample_prefix}_concat.wav"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path, format="wav")

    if verbose:
        print(f"\nExported: {output_path}")
        print(f"Duration: {len(combined) / 1000:.2f}s")
        print(f"Scales traveled: {' -> '.join(j['scale'] for j in journey)}")

    return output_path, journey


def generate_companion_midi(
    journey: List[Dict],
    scales_data: Dict,
    output_path: Path,
    instrument_name: str = "Cello",
    velocity: int = 100,
    base_octave: int = 60,
    use_transposed_pitches: bool = True,
) -> Path:
    """Generate a companion MIDI file precisely timed to audio chunks.

    Args:
        journey: List of measure info dicts from concatenate_audio
        scales_data: Loaded scales dictionary
        output_path: Output MIDI path
        instrument_name: MIDI instrument name
        velocity: Note velocity (0-127)
        base_octave: Base MIDI note number (60 = middle C)
        use_transposed_pitches: If True, use transposed sample pitches; if False, use scale pitches

    Returns:
        Path to generated MIDI file
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi required for MIDI generation: pip install pretty_midi")

    midi = pretty_midi.PrettyMIDI()
    program = pretty_midi.instrument_name_to_program(instrument_name)
    instrument = pretty_midi.Instrument(program=program)

    for entry in journey:
        start_sec = entry["start_ms"] / 1000.0
        duration_sec = entry["duration_ms"] / 1000.0
        end_sec = start_sec + duration_sec

        # Use transposed pitch classes from the actual sample
        if use_transposed_pitches and entry.get("transposed_pitch_classes"):
            pitch_classes = entry["transposed_pitch_classes"]
        else:
            # Fallback to scale pitch classes
            pitch_classes = entry.get("scale_pitch_classes", [])

        # Add notes for each pitch
        for pc in pitch_classes:
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=base_octave + pc,
                start=start_sec,
                end=end_sec,
            )
            instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(str(output_path))

    return output_path


def generate_score_report(
    journey: List[Dict],
    output_path: Path,
    title: str = "Audio Concatenation Score",
) -> Path:
    """Generate a markdown report showing the compositional journey.

    Args:
        journey: List of measure info dicts from concatenate_audio
        output_path: Output markdown file path
        title: Report title

    Returns:
        Path to generated report
    """
    lines = [
        f"# {title}",
        "",
        f"**Total measures:** {len(journey)}",
        f"**Total duration:** {sum(j['duration_ms'] for j in journey) / 1000:.2f}s",
        "",
        "---",
        "",
    ]

    for entry in journey:
        measure = entry["measure"]
        sample = entry["sample"]
        scale = entry["scale"]
        transposition = entry["transposition"]
        duration_ms = entry["duration_ms"]
        start_ms = entry.get("start_ms", 0)

        original_pcs = entry.get("original_pitch_classes", [])
        transposed_pcs = entry.get("transposed_pitch_classes", [])
        scale_pcs = entry.get("scale_pitch_classes", [])

        # Convert pitch classes to note names for readability
        original_notes = [pitch_class_to_note_name(pc) for pc in original_pcs]
        transposed_notes = [pitch_class_to_note_name(pc) for pc in transposed_pcs]
        scale_notes = [pitch_class_to_note_name(pc) for pc in scale_pcs]

        trans_str = f"+{transposition}" if transposition > 0 else str(transposition)
        if transposition == 0:
            trans_str = "0 (original)"

        lines.extend([
            f"## Measure {measure}",
            "",
            f"**Sample:** `{sample}`",
            f"**Time:** {start_ms/1000:.2f}s - {(start_ms + duration_ms)/1000:.2f}s ({duration_ms}ms)",
            f"**Scale:** {scale}",
            f"**Transposition:** {trans_str} semitones",
            "",
            "| | Pitch Classes | Notes |",
            "|---|---|---|",
            f"| Original | {original_pcs} | {', '.join(original_notes)} |",
            f"| Transposed | {transposed_pcs} | {', '.join(transposed_notes)} |",
            f"| Scale | {scale_pcs} | {', '.join(scale_notes)} |",
            "",
            "---",
            "",
        ])

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path
