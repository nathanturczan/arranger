# Arranger

Musical arrangement tools for chaining audio samples and MIDI into continuous compositions based on harmonic compatibility using the Tymoczko 57-scale network.

## Overview

Arranger provides a complete pipeline for creating harmonically-coherent audio compositions:

1. **Sample Chain Generation**: Builds chains of audio samples that transition smoothly based on pitch-class compatibility and voice-leading
2. **Reactive Transposition**: Real-time pitch adjustment with smooth glissando transitions as harmonic context changes
3. **Multi-Layer Rendering**: Stack multiple instrument layers (bass flute, organetta, feedback, etc.) with independent timing and overlap controls
4. **MIDI Export**: Generates companion MIDI files for voicing and bass parts

**Key Features:**
- NumPy-accelerated audio rendering (10-100x faster than pydub-based processing)
- Output-time scheduling for precise timing alignment
- Multi-bit-depth audio support (8/16/24/32-bit samples)
- Configurable layer intervals and overlap limits
- Tymoczko 57-scale network for harmonic navigation

## Installation

```bash
# Clone the repository
git clone https://github.com/nathanturczan/arranger.git
cd arranger

# Install dependencies
pip install -e ".[audio]"
```

### Dependencies
- Python 3.9+
- numpy >= 1.20.0
- pydub >= 0.25.0
- pretty_midi >= 0.2.10
- music21 >= 9.0.0

---

## Main Script: chain_quadruple_hierarchy.py

The primary tool for generating arrangements.

### Quick Start

```bash
# Generate a 30-second arrangement with organetta and feedback layers
python scripts/chain_quadruple_hierarchy.py --seed 789 --organetta --feedback --duration 30

# Full render with all layers
python scripts/chain_quadruple_hierarchy.py --seed 42 --organetta --feedback --duration 120
```

### Usage

```bash
python scripts/chain_quadruple_hierarchy.py [options]

Layer Options:
  --organetta           Enable organetta layer (every 4s)
  --feedback            Enable feedback layer (every 4s, max 2 overlap)
  --no-bassflute        Disable bass flute layer
  --no-midi-synth       Disable MIDI synth rendering

Timing Options:
  --duration SECONDS    Maximum duration in seconds
  --feedback-interval   Feedback interval in seconds (default: 4)

Other Options:
  --seed SEED           Random seed for reproducibility
  --no-glaz-sax         Disable Glaz/Sax samples
  --no-hyacinthe        Disable Hyacinthe samples
  --no-kraus            Disable Kraus samples
```

### Output

The script generates:
- `output/feldman_s<seed>_midi_bf_<timestamp>.wav` - Main audio file
- `output/feldman_s<seed>_midi_bf_<timestamp>_voicing.mid` - Chord voicings MIDI
- `output/feldman_s<seed>_midi_bf_<timestamp>_bass.mid` - Bass line MIDI
- `output/feldman_s<seed>_midi_bf_<timestamp>.json` - Chain metadata

---

## Architecture

### Reactive Transposition

Samples are pitch-shifted in real-time to match the current harmonic context:

1. **Initial Fit**: At sample onset, find the transposition that maps sample pitch classes into the current chord
2. **Event Detection**: Monitor harmonic timeline for chord changes during playback
3. **Glissando**: Smooth pitch transitions (150ms quarter-sine ease) when chord changes
4. **Output-Time Scheduling**: All events scheduled in rendered time for precise alignment

### Layer System

Each layer has independent controls:
- **Bass Flute**: Continuous playback, samples chain back-to-back
- **Organetta**: Interval-based (every 4s), sequential sample selection
- **Feedback**: Interval-based (every 4s), random selection, max 2 overlap

### Audio Processing

- **Caching**: Samples loaded once, stored as normalized float32 NumPy arrays
- **Multi-bit Depth**: Automatic normalization for 8/16/24/32-bit samples
- **Vectorized Rendering**: NumPy-based interpolation, no per-sample loops

---

## Legacy Scripts

### MIDI Concatenation

```bash
python scripts/concatenate_midi.py <start_file> -n 20 --no-play
```

Analyzes chord progressions and builds a directed graph for random walks.

### Audio Concatenation

```bash
python scripts/concatenate_audio.py organize /path/to/samples samples_data.json
python scripts/concatenate_audio.py chain /path/to/scales_dir prefix -n 50
```

Basic scale-based sample chaining.

---

## Datasets

The `datasets/` directory contains ~905 MIDI files from various music theory sources:

| Dataset | Files | Source |
|---------|-------|--------|
| schoenberg | 198 | Harmonielehre |
| reger | 121 | Modulation treatise |
| boyd-jazz | 114 | Jazz harmony |
| messiaen | 101 | Modes of limited transposition |
| hindemith | 65 | Craft of Musical Composition |
| persichetti | 37 | Twentieth-Century Harmony |

See `datasets/README.md` for full provenance documentation.

---

## References

- Tymoczko, D. (2011). *A Geometry of Music*
- Turczan, N. (2019). Scale Navigator: A Networked Approach to Scale-Based Composition. NIME 2019.

## License

MIT License - See LICENSE file for details.
