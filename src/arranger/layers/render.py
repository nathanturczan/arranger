"""
Unified layer rendering.

All layers go through render_layer() - no custom rendering functions per layer.
This ensures consistent behavior (NumPy rendering, same gain handling, same transposition).
"""

import numpy as np
from typing import List, Callable, Optional, TYPE_CHECKING

from .config import LayerConfig, LayerType, SelectionMode

if TYPE_CHECKING:
    from dataclasses import dataclass


def render_layer(
    config: LayerConfig,
    harmonic_start_times: List[float],
    harmonic_events: List,  # List[HarmonicEvent]
    find_transposition_fn: Callable,
    total_duration_ms: float,
    sample_rate: int = 44100,
    channels: int = 2,
    verbose: bool = False,
    seed: Optional[int] = None,
    # These are the actual rendering functions - passed in to avoid circular imports
    render_continuous_fn: Callable = None,
    render_interval_fn: Callable = None,
    render_cloud_fn: Callable = None,
    render_chord_triggered_fn: Callable = None,
) -> np.ndarray:
    """
    Render a layer using its configuration.

    This is the ONLY entry point for rendering layers. All layers use the same
    underlying NumPy rendering functions, just with different parameters.

    Args:
        config: LayerConfig defining the layer's behavior
        harmonic_start_times: Sorted list of harmonic event start times
        harmonic_events: List of HarmonicEvent objects
        find_transposition_fn: Function to find pitch transposition
        total_duration_ms: Total duration to render
        sample_rate: Audio sample rate
        channels: Number of audio channels
        verbose: Print progress info
        seed: Random seed for reproducibility
        render_continuous_fn: The render_layer_continuous_np function
        render_interval_fn: The render_layer_interval_np function
        render_cloud_fn: The render_layer_cloud_np function

    Returns:
        NumPy float32 buffer [total_samples, channels]
    """
    if not config.enabled:
        total_samples = int(total_duration_ms * sample_rate / 1000)
        return np.zeros((total_samples, channels), dtype=np.float32)

    if not config.samples:
        if verbose:
            print(f"    {config.name}: No samples loaded, skipping")
        total_samples = int(total_duration_ms * sample_rate / 1000)
        return np.zeros((total_samples, channels), dtype=np.float32)

    if config.layer_type == LayerType.CONTINUOUS:
        if render_continuous_fn is None:
            raise ValueError("render_continuous_fn must be provided for CONTINUOUS layers")

        return render_continuous_fn(
            sample_list=config.samples,
            harmonic_start_times=harmonic_start_times,
            harmonic_events=harmonic_events,
            find_transposition_fn=find_transposition_fn,
            total_duration_ms=total_duration_ms,
            sample_rate=sample_rate,
            channels=channels,
            gain_db=config.gain_db,
            verbose=verbose,
            layer_name=config.name,
        )

    elif config.layer_type == LayerType.INTERVAL:
        if render_interval_fn is None:
            raise ValueError("render_interval_fn must be provided for INTERVAL layers")

        return render_interval_fn(
            sample_list=config.samples,
            harmonic_start_times=harmonic_start_times,
            harmonic_events=harmonic_events,
            find_transposition_fn=find_transposition_fn,
            total_duration_ms=total_duration_ms,
            interval_seconds=config.interval_seconds,
            sample_rate=sample_rate,
            channels=channels,
            gain_db=config.gain_db,
            verbose=verbose,
            layer_name=config.name,
            skip_to_fitting=True,
            random_selection=(config.selection == SelectionMode.RANDOM),
            seed=seed,
            max_overlap=config.max_overlap,
        )

    elif config.layer_type == LayerType.CLOUD:
        if render_cloud_fn is None:
            raise ValueError("render_cloud_fn must be provided for CLOUD layers")

        return render_cloud_fn(
            sample_list=config.samples,
            harmonic_start_times=harmonic_start_times,
            harmonic_events=harmonic_events,
            find_transposition_fn=find_transposition_fn,
            total_duration_ms=total_duration_ms,
            min_silence_seconds=config.min_silence_seconds,
            max_silence_seconds=config.max_silence_seconds,
            cloud_duration_seconds=config.cloud_duration_seconds,
            bpm=config.bpm,
            note_division=config.note_division,
            sample_rate=sample_rate,
            channels=channels,
            gain_db=config.gain_db,
            verbose=verbose,
            layer_name=config.name,
            random_selection=(config.selection == SelectionMode.RANDOM),
            seed=seed,
        )

    elif config.layer_type == LayerType.CHORD_TRIGGERED:
        if render_chord_triggered_fn is None:
            raise ValueError("render_chord_triggered_fn must be provided for CHORD_TRIGGERED layers")

        return render_chord_triggered_fn(
            sample_list=config.samples,
            harmonic_start_times=harmonic_start_times,
            harmonic_events=harmonic_events,
            find_transposition_fn=find_transposition_fn,
            total_duration_ms=total_duration_ms,
            sample_rate=sample_rate,
            channels=channels,
            gain_db=config.gain_db,
            verbose=verbose,
            layer_name=config.name,
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown layer type: {config.layer_type}")
