"""Tymoczko 57-scale network operations.

The 57-scale network provides a graph of scales connected by minimal
voice-leading distance. This enables smooth harmonic transitions in
both MIDI and audio concatenation.
"""

from .tymoczko import (
    note_name_to_pitch_class,
    pitch_class_to_note_name,
    transpose_pitch_classes,
    is_subset_of_scale,
    load_scales_data,
    get_adjacent_scales,
    get_scale_pitch_classes,
    find_compatible_scales,
    get_scale_family,
)

__all__ = [
    "note_name_to_pitch_class",
    "pitch_class_to_note_name",
    "transpose_pitch_classes",
    "is_subset_of_scale",
    "load_scales_data",
    "get_adjacent_scales",
    "get_scale_pitch_classes",
    "find_compatible_scales",
    "get_scale_family",
]
