# Concatenation Approaches Master Tracker

## Overview

This document tracks different approaches to musical concatenation—chaining audio snippets, MIDI progressions, or notation fragments end-to-end based on harmonic compatibility.

---

## Approach Comparison

| Approach | Domain | Matching Logic | Transposition Method | Scale Network | Status |
|----------|--------|----------------|---------------------|---------------|--------|
| **Progression Concatenator** | MIDI | Voice-leading intervals (4-6 voice flexible matching) | Bass note alignment + octave correction | Graph of compatible progressions | **Active** |
| **Sample Laboratory** | Audio | Pitch class subset → scale membership | Pre-computed transpositions (-7 to +4 semitones) | Tymoczko 57-scale adjacency | Active |
| **Reger Mods Chainer** | MIDI/LilyPond | First/last chord matching | Transpose entire progression | TBD | Planned |
| **Sarabande Project** | Audio | Pitch class subset → scale membership | Scale network navigation | Tymoczko 57-scale adjacency | Planned |

---

## Source Material Inventory

### Audio Sample Collections

| Collection | Source | Samples | Tagged | Location | Status |
|------------|--------|---------|--------|----------|--------|
| **Whiteroom** | Cream - White Room | 9 triads | Yes | `/Users/soney/Music/sample-laboratory/whiteroom/` | Complete |
| **RHCP** | Red Hot Chili Peppers | 15 | Yes | `/Users/soney/Music/sample-laboratory/rhcp/` | Complete |
| **Mayer** | John Mayer | 9 | Yes | `/Users/soney/Music/sample-laboratory/mayer/` | Complete |
| **Wild Rose** | MacDowell - To a Wild Rose | 50 | Yes | `/Users/soney/Music/sample-laboratory/wildrose/` | Complete |
| **Mamamia** | ABBA - Mamma Mia | Multiple | Yes | `/Users/soney/Music/sample-laboratory/mamamia/` | Complete |
| **ILYAF** | I Love You Always Forever | Multiple | Yes | `/Users/soney/Music/sample-laboratory/ilyaf/` | Complete |
| **Sarabande** | Satie - Sarabande | 78 | Yes | `/Users/soney/Music/samples/sarabande/` | Ready for processing |

### MIDI/Notation Collections

**Consolidated Location:** `/Users/soney/Github/concatenator/datasets/`

| Collection | Source | Items | Format | Status |
|------------|--------|-------|--------|--------|
| **Schoenberg** | Harmonielehre examples | 198 | MIDI | Active |
| **Reger** | Max Reger modulation exercises | 121 | MIDI + LilyPond | Active |
| **Boyd-Jazz** | Jazz chord progressions | 114 | MIDI | Active |
| **Messiaen** | Modes of limited transposition | 101 | MIDI | Active |
| **Misc** | Various sources | 74 | MIDI | Active |
| **Hindemith** | Craft of Musical Composition | 65 | MIDI | Active |
| **HAM** | Harmonic Analysis Materials | 52 | MIDI | Active |
| **Hanson** | Harmonic Materials of Modern Music | 52 | MIDI | Active |
| **Levarie** | Theory textbook | 39 | MIDI | Active |
| **Persichetti** | Twentieth-Century Harmony | 37 | MIDI | Active |
| **Slonimsky** | Thesaurus of Scales | 36 | MIDI + LilyPond | Active |
| **VocalArr** | Vocal arrangements | 11 | MIDI | Active |
| **NAKH** | Miscellaneous | 4 | MIDI | Active |
| **Wagner** | Wagner excerpts | 2 | MIDI | Active |
| **Wischnegradsky** | Microtonal examples | 1 | MIDI | Active |

---

## Detailed Approach Documentation

### 1. Sample Laboratory (Audio Chaining)

**Location:** `/Users/soney/Music/sample-laboratory/`

**Core Files:**
- `sample_clerk.py` - Organizes samples into scale folders via symlinks
- `scales_data.json` - 57 scales with adjacency relationships
- `{collection}/make_song.py` - Chains samples via adjacent scale navigation

**Algorithm:**
1. Tag each sample with its pitch classes (note_names in JSON)
2. Create transpositions from -7 to +4 semitones
3. Distribute samples/transpositions to all compatible scale folders
4. Chain by walking through adjacent scales, selecting next numbered sample
5. Concatenate audio with 1ms crossfade

**Scales Supported:** 57 scales
- 12 Diatonic, 12 Acoustic, 12 Harmonic minor, 12 Harmonic major
- 3 Octatonic, 2 Whole-tone, 4 Hexatonic

---

### 2. Progression Concatenator (MIDI Chaining)

**Active Implementation:** `/Users/soney/Github/concatenator/` (consolidated repo)
**Documentation:** `/Users/soney/Github/concatenator/README.md`
**Original Source:** `/Users/soney/Music/MIDI_Experiments/Modulation/progression5.py`

**MIDI Corpus (~905 progressions):**
| Dataset | Count | Source |
|---------|-------|--------|
| Schoenberg | 198 | *Harmonielehre*, pp. 89-93, 360-361 |
| Reger | 121 | Modulation treatise |
| Boyd-Jazz | 114 | Jazz harmony |
| Messiaen | 101 | Modes of limited transposition |
| Misc | 74 | Various sources |
| Hindemith | 65 | *Craft of Musical Composition* |
| HAM | 52 | Harmonic Analysis Materials |
| Hanson | 52 | *Harmonic Materials of Modern Music* |
| Levarie | 39 | Theory textbook |
| Persichetti | 37 | *Twentieth-Century Harmony* |
| Slonimsky | 36 | *Thesaurus of Scales* |
| VocalArr | 11 | Vocal arrangements |
| NAKH | 4 | Miscellaneous |
| Wagner | 2 | Wagner excerpts |
| Wischnegradsky | 1 | Microtonal examples |

**Algorithm:**
1. Parse MIDI progressions using music21
2. Extract SATB voicing with flexible 4-6 voice matching via sliding window
3. Build directed graph: progression A → B if any 4-note window intervals match
4. Prune sinks iteratively (progressions with no valid children)
5. Random walk through graph, biasing toward loop closure after target length
6. Apply octave correction for voice leading continuity
7. Add ties for repeated notes, export to LilyPond (default) or MusicXML

**Usage:**
```bash
cd /Users/soney/Github/concatenator
python scripts/concatenate_midi.py datasets/schoenberg/harmonielehre-p89-91-inversions-of-7th-chords-1.midi -n 100 --no-play
```

**Output:** LilyPond file (default) or MusicXML score, with optional realtime MIDI playback

---

### 3. Reger Modulations Chainer (Planned)

**Source Material:**
- 100+ modulations starting from C major
- Each ends in a different key
- 4-part harmony (SATB)
- a/b variants for enharmonic alternatives

**Planned Approach:**
- Analyze ending chord of each modulation
- Match to starting chord (C major) of next modulation
- Transpose entire modulation to create smooth transitions
- Support both sequential playback and graph-based navigation

---

### 4. Sarabande Project (Planned)

**Source Material:**
- Satie Sarabande broken into ~80 chord segments
- Each segment tagged with pitch classes
- Similar structure to Wild Rose project

**Planned Approach:**
- Copy sample-laboratory workflow
- Create scales_dir with symlinked transpositions
- Chain via adjacent scale navigation
- Output: continuously modulating Satie

---

## Technical Components

### Shared Dependencies

```
pydub          # Audio concatenation
music21        # MIDI parsing/analysis
pretty_midi    # MIDI generation
librosa        # Audio analysis
graphviz       # Network visualization
```

### Scale Network (Tymoczko)

The 57-scale network is based on Dmitri Tymoczko's research on voice leading and scale relationships. Adjacent scales share maximum pitch class overlap, enabling smooth modulation.

**Key Properties:**
- Each scale has 4-6 adjacent scales
- Adjacency = minimal voice leading distance
- Enables infinite harmonic walks without jarring transitions

---

## Project Roadmap

### Immediate (Sarabande)
- [ ] Copy sample-laboratory structure to sarabande folder
- [ ] Run sample_clerk.py to create scales_dir
- [ ] Create make_song.py for sarabande
- [ ] Generate test outputs

### Near-term (Reger Mods)
- [ ] Analyze all Reger modulations for first/last chords
- [ ] Build compatibility graph
- [ ] Create concatenation script
- [ ] Generate continuous modulation pieces

### Future
- [ ] Unified concatenation tool supporting MIDI + audio
- [ ] Real-time concatenation in Max/MSP or Pure Data
- [ ] Web interface for exploring concatenation possibilities
- [ ] Integration with Scale Navigator ecosystem

---

## References

- Tymoczko, D. (2011). *A Geometry of Music*
- Turczan, N., Kapur, A., Ho, N., Honigman, C., & Shepherd, D. (2019). The Scale Navigator: A System for Networked Algorithmic Harmony. NIME'19, Porto Alegre, Brazil. https://doi.org/10.5281/zenodo.3672882
- MacDowell's "To a Wild Rose" with random modulations: https://nathanturczan.bandcamp.com
- Ted Gioia tweet: https://twitter.com/tedgioia/status/1321481234567890

---

*Last updated: 2026-02-11*
