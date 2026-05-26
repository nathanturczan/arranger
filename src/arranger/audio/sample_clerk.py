"""Sample organization and transposition for scale-based audio concatenation.

This module handles:
- Organizing audio samples into scale folders via symlinks
- Transposing samples to different keys (-7 to +4 semitones)
- Building a scales_dir structure for efficient lookup during concatenation
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from pydub import AudioSegment, effects

from ..scales.tymoczko import (
    note_name_to_pitch_class,
    transpose_pitch_classes,
    is_subset_of_scale,
    load_scales_data,
)


# Default transposition range (in semitones)
DEFAULT_TRANSPOSITIONS = (-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4)


def transpose_audio(
    input_path: Path,
    semitones: int,
    output_path: Optional[Path] = None,
    target_sample_rate: int = 44100,
) -> Path:
    """Transpose an audio file by changing its playback rate.

    This uses sample rate manipulation for pitch shifting, which also
    affects duration. For cleaner results, consider using a proper
    pitch shifting algorithm.

    Args:
        input_path: Path to input WAV file
        semitones: Number of semitones to transpose (positive or negative)
        output_path: Path for output file (auto-generated if None)
        target_sample_rate: Target sample rate for output

    Returns:
        Path to the transposed audio file
    """
    if output_path is None:
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.parent / f"{stem}_transposed_by{semitones:+d}{suffix}"

    sound = AudioSegment.from_file(input_path, format="wav")

    # Calculate new sample rate for pitch shift
    new_sample_rate = int(sound.frame_rate * (2 ** (semitones / 12)))

    # Resample to shift pitch
    transposed = sound._spawn(
        sound.raw_data,
        overrides={'frame_rate': new_sample_rate}
    )
    transposed = transposed.set_frame_rate(target_sample_rate)

    transposed.export(output_path, format='wav')
    return output_path


def normalize_audio(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """Normalize audio file to consistent loudness.

    Args:
        input_path: Path to input WAV file
        output_path: Path for output (overwrites input if None)

    Returns:
        Path to normalized file
    """
    if output_path is None:
        output_path = input_path

    raw = AudioSegment.from_file(input_path, "wav")
    normalized = effects.normalize(raw)
    normalized.export(output_path, format="wav")
    return output_path


def load_samples_manifest(path: Path) -> Dict:
    """Load the samples manifest JSON.

    The manifest should have structure:
    {
        "sample_name": {
            "note_names": ["C", "E", "G"],
            "pitch_classes": [0, 4, 7]  # Optional, will be computed if missing
        },
        ...
    }

    Args:
        path: Path to samples_data.json

    Returns:
        Dictionary of sample metadata
    """
    with open(path) as f:
        samples_dict = json.load(f)

    # Compute pitch classes if missing
    for sample_name, sample_info in samples_dict.items():
        if 'pitch_classes' not in sample_info:
            sample_info['pitch_classes'] = [
                note_name_to_pitch_class(note)
                for note in sample_info.get('note_names', [])
            ]

    return samples_dict


def organize_samples(
    samples_dir: Path,
    samples_manifest: Path,
    scales_data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    transpositions: tuple = DEFAULT_TRANSPOSITIONS,
    use_symlinks: bool = True,
    verbose: bool = True,
) -> Path:
    """Organize samples into scale folders for efficient concatenation lookup.

    This creates a scales_dir/ structure where each scale folder contains
    symlinks (or copies) to all samples (and their transpositions) that
    fit within that scale.

    Args:
        samples_dir: Directory containing source WAV samples
        samples_manifest: Path to samples_data.json with pitch class info
        scales_data_path: Path to scales_data.json (uses default if None)
        output_dir: Output directory for scales_dir (defaults to samples_dir/scales_dir)
        transpositions: Tuple of semitone transpositions to generate
        use_symlinks: If True, use symlinks; if False, copy files
        verbose: Print progress messages

    Returns:
        Path to the created scales_dir
    """
    scales_data = load_scales_data(scales_data_path)
    samples_dict = load_samples_manifest(samples_manifest)

    if output_dir is None:
        output_dir = samples_dir / "scales_dir"

    # Create scale folders
    for scale_name in scales_data:
        (output_dir / scale_name).mkdir(parents=True, exist_ok=True)

    # Process each sample
    for sample_name, sample_info in samples_dict.items():
        sample_path = samples_dir / f"{sample_name}.wav"

        if not sample_path.exists():
            if verbose:
                print(f"Warning: Sample not found: {sample_path}")
            continue

        pitch_classes = sample_info.get('pitch_classes', [])

        if not pitch_classes:
            if verbose:
                print(f"Skipping {sample_name}: no pitch class info")
            continue

        # Link/copy original sample to compatible scales
        _distribute_sample(
            sample_path, pitch_classes, scales_data, output_dir,
            use_symlinks, verbose
        )

        # Generate and distribute transpositions
        for semitones in transpositions:
            transposed_path = transpose_audio(sample_path, semitones)
            transposed_pcs = transpose_pitch_classes(pitch_classes, semitones)

            _distribute_sample(
                transposed_path, transposed_pcs, scales_data, output_dir,
                use_symlinks, verbose
            )

    return output_dir


def _distribute_sample(
    sample_path: Path,
    pitch_classes: List[int],
    scales_data: Dict,
    output_dir: Path,
    use_symlinks: bool,
    verbose: bool,
):
    """Distribute a sample to all compatible scale folders."""
    for scale_name, scale_info in scales_data.items():
        scale_pcs = scale_info.get('pitch_classes', [])

        if is_subset_of_scale(scale_pcs, pitch_classes):
            dest = output_dir / scale_name / sample_path.name

            if dest.exists():
                continue

            if use_symlinks:
                if verbose:
                    print(f"Linking {sample_path.name} -> {scale_name}")
                dest.symlink_to(sample_path.resolve())
            else:
                if verbose:
                    print(f"Copying {sample_path.name} -> {scale_name}")
                shutil.copy2(sample_path, dest)
