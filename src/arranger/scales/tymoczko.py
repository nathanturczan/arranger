"""Tymoczko 57-scale network operations.

The 57-scale network is based on Dmitri Tymoczko's research on voice leading
and scale relationships. Adjacent scales share maximum pitch class overlap,
enabling smooth modulation.

Scale families:
- 12 Diatonic (major modes)
- 12 Acoustic (melodic minor modes)
- 12 Harmonic minor
- 12 Harmonic major
- 3 Octatonic (diminished)
- 2 Whole-tone
- 4 Hexatonic (augmented)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set


# Pitch name to pitch class mapping
PITCH_MAP = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
ACCIDENTAL_MAP = {'#': 1, 's': 1, '': 0, '♭': -1, 'f': -1, '-': -1, '!': 0}


def note_name_to_pitch_class(note_name: str) -> int:
    """Convert a note name to its pitch class (0-11).

    Supports various accidental notations:
    - Sharp: # or s (e.g., 'C#', 'Cs')
    - Flat: b, f, ♭, or - (e.g., 'Bb', 'Bf', 'B♭', 'B-')

    Args:
        note_name: Note name like 'C', 'C#', 'Bb', 'Fs'

    Returns:
        Pitch class as integer 0-11 (C=0, C#=1, ..., B=11)

    Raises:
        ValueError: If note name format is invalid
    """
    try:
        match = re.match(r'^(?P<n>[A-Ga-g])(?P<off>[#s♭fb!\-]?)$', note_name)
        pitch = match.group('n').upper()
        offset = ACCIDENTAL_MAP.get(match.group('off'), 0)
    except (AttributeError, KeyError):
        raise ValueError(f'Invalid note format: {note_name}')

    return (PITCH_MAP[pitch] + offset) % 12


def pitch_class_to_note_name(pitch_class: int, prefer_sharp: bool = True) -> str:
    """Convert a pitch class to a note name.

    Args:
        pitch_class: Integer 0-11
        prefer_sharp: If True, use sharps; if False, use flats

    Returns:
        Note name string
    """
    if prefer_sharp:
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    else:
        names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    return names[pitch_class % 12]


def transpose_pitch_classes(pitch_classes: List[int], semitones: int) -> List[int]:
    """Transpose a list of pitch classes by a given interval.

    Args:
        pitch_classes: List of pitch classes (0-11)
        semitones: Number of semitones to transpose (can be negative)

    Returns:
        New list of transposed pitch classes
    """
    return [(pc + semitones) % 12 for pc in pitch_classes]


def is_subset_of_scale(scale_pcs: List[int], sample_pcs: List[int]) -> bool:
    """Check if sample pitch classes are a subset of scale pitch classes.

    Args:
        scale_pcs: List of pitch classes in the scale
        sample_pcs: List of pitch classes in the sample

    Returns:
        True if all sample pitch classes are in the scale
    """
    return set(sample_pcs).issubset(set(scale_pcs))


def load_scales_data(path: Optional[Path] = None) -> Dict:
    """Load the 57-scale network data from JSON.

    Args:
        path: Path to scales_data.json. If None, uses default location.

    Returns:
        Dictionary with scale definitions and adjacencies
    """
    if path is None:
        # Default to the data directory in the package
        path = Path(__file__).parent.parent.parent.parent / "data" / "scales_data.json"

    with open(path) as f:
        return json.load(f)


def get_adjacent_scales(scales_data: Dict, scale_name: str) -> List[str]:
    """Get the adjacent scales for a given scale.

    Args:
        scales_data: Loaded scales dictionary
        scale_name: Name of the scale

    Returns:
        List of adjacent scale names
    """
    if scale_name not in scales_data:
        raise ValueError(f"Unknown scale: {scale_name}")
    return scales_data[scale_name].get('adjacent_scales', [])


def get_scale_pitch_classes(scales_data: Dict, scale_name: str) -> List[int]:
    """Get the pitch classes for a given scale.

    Args:
        scales_data: Loaded scales dictionary
        scale_name: Name of the scale

    Returns:
        List of pitch classes in the scale
    """
    if scale_name not in scales_data:
        raise ValueError(f"Unknown scale: {scale_name}")
    return scales_data[scale_name].get('pitch_classes', [])


def find_compatible_scales(scales_data: Dict, pitch_classes: List[int]) -> List[str]:
    """Find all scales that contain the given pitch classes as a subset.

    Args:
        scales_data: Loaded scales dictionary
        pitch_classes: List of pitch classes to match

    Returns:
        List of compatible scale names
    """
    compatible = []
    for scale_name, scale_info in scales_data.items():
        scale_pcs = scale_info.get('pitch_classes', [])
        if is_subset_of_scale(scale_pcs, pitch_classes):
            compatible.append(scale_name)
    return compatible


def get_scale_family(scales_data: Dict, scale_name: str) -> str:
    """Get the family/class of a scale (diatonic, acoustic, etc.).

    Args:
        scales_data: Loaded scales dictionary
        scale_name: Name of the scale

    Returns:
        Scale family name
    """
    if scale_name not in scales_data:
        raise ValueError(f"Unknown scale: {scale_name}")
    return scales_data[scale_name].get('scale_class', 'unknown')
