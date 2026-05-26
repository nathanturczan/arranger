"""Audio concatenation via scale-based navigation.

This module provides tools for:
- Organizing audio samples into scale folders (sample_clerk)
- Chaining samples via the Tymoczko 57-scale network (chain)
- Generating companion MIDI tracks
"""

from .sample_clerk import (
    organize_samples,
    transpose_audio,
    normalize_audio,
    load_samples_manifest,
)

from .chain import (
    concatenate_audio,
    find_initial_samples,
    find_next_samples,
    generate_companion_midi,
    generate_score_report,
)

__all__ = [
    # sample_clerk
    "organize_samples",
    "transpose_audio",
    "normalize_audio",
    "load_samples_manifest",
    # chain
    "concatenate_audio",
    "find_initial_samples",
    "find_next_samples",
    "generate_companion_midi",
    "generate_score_report",
]
