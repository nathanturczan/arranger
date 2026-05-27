"""
Layer configuration dataclass.

All layers are defined by their configuration, not by custom rendering functions.
This ensures consistent behavior across all layer types.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any


class LayerType(Enum):
    """How samples are scheduled in time."""
    CONTINUOUS = "continuous"  # Back-to-back, one at a time (bass flute, jicello)
    INTERVAL = "interval"      # Fire every N seconds (organetta, feedback, prophet)
    CLOUD = "cloud"            # Bursts of rapid notes with silence between (brodero, gothicharp)
    CHORD_TRIGGERED = "chord_triggered"  # Fires on chord changes if fitting sample exists (godette)


class SelectionMode(Enum):
    """How samples are selected from the pool."""
    SEQUENTIAL = "sequential"  # Round-robin through samples in order
    RANDOM = "random"          # Random selection from pool


@dataclass
class LayerConfig:
    """
    Configuration for a sample layer.

    All layers use the same rendering pipeline - they differ only in configuration.
    This prevents inconsistent behavior between layers.

    Attributes:
        name: Display name for logging
        samples: List of sample dicts with 'name', 'pitch_classes', 'audio_path'
        layer_type: CONTINUOUS (back-to-back) or INTERVAL (every N seconds)
        interval_seconds: For INTERVAL layers, time between fires
        max_overlap: For INTERVAL layers, max concurrent samples (None = unlimited)
        gain_db: Volume adjustment in decibels
        selection: SEQUENTIAL or RANDOM sample selection
        enabled: Whether this layer is active
    """
    name: str
    samples: List[Dict]
    layer_type: LayerType

    # Timing (for INTERVAL layers)
    interval_seconds: float = 4.0
    max_overlap: Optional[int] = None

    # Timing (for CLOUD layers)
    min_silence_seconds: float = 10.0  # Minimum silence between clouds
    max_silence_seconds: float = 15.0  # Maximum silence between clouds
    cloud_duration_seconds: float = 4.0  # Duration of each burst
    bpm: float = 120.0  # Tempo for note subdivision
    note_division: int = 16  # 16 for 16th notes, 32 for 32nd notes

    # Audio
    gain_db: float = 0.0

    # Selection
    selection: SelectionMode = SelectionMode.SEQUENTIAL

    # State
    enabled: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.layer_type == LayerType.INTERVAL and self.interval_seconds <= 0:
            raise ValueError(f"Layer {self.name}: interval_seconds must be > 0")
        if self.max_overlap is not None and self.max_overlap < 1:
            raise ValueError(f"Layer {self.name}: max_overlap must be >= 1 or None")
        if self.layer_type == LayerType.CLOUD:
            if self.min_silence_seconds <= 0 or self.max_silence_seconds <= 0:
                raise ValueError(f"Layer {self.name}: silence durations must be > 0")
            if self.min_silence_seconds > self.max_silence_seconds:
                raise ValueError(f"Layer {self.name}: min_silence must be <= max_silence")
            if self.cloud_duration_seconds <= 0:
                raise ValueError(f"Layer {self.name}: cloud_duration_seconds must be > 0")
            if self.note_division not in (4, 8, 16, 32):
                raise ValueError(f"Layer {self.name}: note_division must be 4, 8, 16, or 32")
