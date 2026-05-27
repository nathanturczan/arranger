
---

## Issue: Handel sample out of tune near 35 seconds

**Labels:** `bug`, `audio-quality`

**Description:**
In render `feldman_s789_midi_bf_bro_jic_20260525_234647.wav`, a Handel sample occurring just before the 35-second mark sounds out of tune.

**To investigate:**
1. Check which Handel sample is playing at ~35s (look in JSON output)
2. Verify the transposition calculation is correct
3. Check if the sample's pitch class metadata is accurate
4. Possible causes:
   - Incorrect pitch class detection in original sample
   - Wrong transposition being applied
   - Glissando artifact

**Repro:**
```bash
python scripts/chain_quadruple_hierarchy.py --seed 789 --no-midi-synth --no-glaz-sax --no-hyacinthe --no-kraus --brodero --jicello --organetta --gentleharpsi --feedback --duration 45
```

Listen at ~35 seconds.

---

## Issue: Glissandos sound stepped/quantized

**Labels:** `enhancement`, `audio-quality`

**Description:**
Pitch glissandos between transposition changes sound stepped rather than smooth. Currently using pydub frame_rate manipulation in discrete chunks.

**Current implementation:**
- `apply_glissando()` in chain_quadruple_hierarchy.py
- Uses 2ms chunks with frame_rate changes
- Crossfades between chunks don't fully eliminate stepping

**Proposed fix:**
Replace pydub frame_rate approach with proper pitch-shifting library:
- **pyrubberband** (preferred) - high quality, Rubber Band C++ library
- **librosa.effects.pitch_shift** - phase vocoder, slower but pure Python

**Installation:**
```bash
pip install pyrubberband
# Requires: brew install rubberband (on macOS)
```

**Example implementation:**
```python
import pyrubberband as pyrb
import numpy as np

def smooth_glissando(audio_array, sr, semitones_start, semitones_end, duration_samples):
    """Apply smooth pitch glissando using pyrubberband."""
    # Generate pitch curve
    pitch_curve = np.linspace(semitones_start, semitones_end, duration_samples)
    # pyrb.pitch_shift handles smooth interpolation
    return pyrb.pitch_shift(audio_array, sr, pitch_curve)
```

