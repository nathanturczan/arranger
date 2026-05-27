# Harmonic Systems Architecture Notes

**Date:** 2026-05-27
**Purpose:** Document the relationships between Librarian, Harmonic Engine, and Arranger to prevent duplication and maximize reuse.

> **See also:** `/Users/soney/Github/harmonic-engine/UNIFIED_ARCHITECTURE.md` for the complete 5-system view including NCS and SkeletonPlayer.

---

## Three Systems Overview

### 1. Librarian (`/Users/soney/Github/librarian/`)
**Language:** Python
**Purpose:** Sample library management, pitch detection, and chord-driven sample matching

**Key Components:**
| File | Function |
|------|----------|
| `librarian.py` | Basic Pitch analysis, `samples_data.json` generation |
| `sketchpad_matcher.py` | Chord-driven matching: given a progression, find samples to bridge transitions |
| `progression_analyzer.py` | Analyze chord progressions in samples |
| `chord_parser.py` | Chord database, pitch class utilities |
| `filename_pitch_parser.py` | Extract pitch info from filenames |
| `sample_sequencer.py` | Legacy sequencer |

**Core Data Structures:**
- `samples_data.json` - per-library manifest with pitch classes, first/last chords
- `_library_type` - "progression", "phrase", "drone", etc.

### 2. Harmonic Engine (`/Users/soney/Github/harmonic-engine/`)
**Language:** TypeScript
**Purpose:** MCP server exposing Scale Navigator's harmonic inference to AI assistants

**Key Components:**
| File | Function |
|------|----------|
| `src/index.ts` | MCP server with tools for scale navigation, chord inference |
| `src/data/scales.ts` | 57-scale network, adjacency, path finding |
| `src/data/chords.ts` | Chord analysis and naming |
| `src/utils/pitch-class.ts` | Pitch class utilities (mod12, transpose) |
| `src/ncs-tools.ts` | NotesChordScales inference engine |

**MCP Tools Exposed:**
- `explore_scale_network` - Navigate scale adjacencies
- `infer_chords_from_notes` - Notes → possible chords
- `infer_scales_from_chord` - Chord → compatible scales
- `find_samples_for_transition` - Sample matching (bridge between systems!)

### 3. Arranger (`/Users/soney/Github/arranger/`)
**Language:** Python
**Purpose:** Sample-driven chain generation with reactive transposition and multi-layer rendering

**Key Components:**
| File | Function |
|------|----------|
| `scripts/chain_quadruple_hierarchy.py` | Main chain builder, layer renderer |
| `src/arranger/samples/registry.py` | Sample library paths |
| `src/arranger/scales/tymoczko.py` | 57-scale definitions (Python port) |
| `data/chords_no_supersets.json` | Chord dictionary |

**Core Algorithms:**
- **Quadruple Hierarchy:** Sample endpoints ⊂ Inferred Chord ⊂ Scale ⊂ Scale Class
- **Reactive Transposition:** Real-time pitch adjustment with glissando

---

## Three Paradigms Compared

| Aspect | sketchpad_matcher (Librarian) | alpine-lady_transitions.json | chain_quadruple_hierarchy (Arranger) |
|--------|-------------------------------|------------------------------|--------------------------------------|
| **Input** | Pre-defined chord progression | N/A (output file) | Sample pool |
| **Output** | Sample matches per transition | N/A (data file) | Audio + MIDI + JSON |
| **Direction** | Chords → Samples | N/A | Samples → Chords → Audio |
| **Control** | Composer defines harmony | N/A | Harmony emerges |
| **Use Case** | "Find samples for my song" | "Store matches for reuse" | "Generate something new" |

### The Key Insight

**sketchpad_matcher** and **arranger** are the same algorithm running in opposite directions:

```
CHORD-DRIVEN (sketchpad_matcher):
  Progression [CM7 → FM7#11 → Em7 → ...]
  → For each transition, find samples whose endpoints fit
  → Output: ranked sample matches per edge

SAMPLE-DRIVEN (arranger):
  Sample pool [feldman, handel, glaz_sax, ...]
  → Current sample's last_pcs → infer bridging chord
  → Find next sample whose first_pcs fits that chord
  → Output: continuous chain with inferred harmony
```

**alpine-lady_transitions.json** is the **cached output** of sketchpad_matcher - a lookup table of which samples can bridge which transitions in your song structure.

---

## Shared Utilities (Currently Duplicated)

### Pitch Class Operations
| Operation | Librarian | Harmonic Engine | Arranger |
|-----------|-----------|-----------------|----------|
| mod12 | `chord_parser.py` | `pitch-class.ts` | inline |
| transpose | `chord_parser.py` | `transposition.ts` | `transpose_pcs()` |
| pc_to_name | `chord_parser.py` | `pitch-class.ts` | `PC_TO_NOTE` dict |
| overlap score | `chord_parser.py` | `index.ts` | `compute_overlap()` |

### Scale/Chord Data
| Data | Librarian | Harmonic Engine | Arranger |
|------|-----------|-----------------|----------|
| 57 scales | `data/pressing_scale_dict.json` | `data/scales.ts` | `tymoczko.py` |
| Chord dictionary | `data/chords_no_supersets.json` | `data/chords.ts` | `data/chords_no_supersets.json` |

### Sample Loading
| Operation | Librarian | Harmonic Engine | Arranger |
|-----------|-----------|-----------------|----------|
| Load manifest | `sketchpad_matcher.py` | `loadSampleLibrary()` | `load_all_samples()` |
| Library paths | scattered | `SAMPLE_LIBRARY_PATHS` | `registry.py` |

---

## Proposed Organization

### Option A: Shared Python Package (`snaps-core`)
Create a shared package for Python projects:

```
snaps-core/
├── pitch_class.py      # mod12, transpose, pc_to_name
├── chords.py           # chord dictionary, inference
├── scales.py           # 57-scale network
├── samples.py          # manifest loading, library discovery
└── matching.py         # overlap scoring, transposition search
```

**Librarian** and **Arranger** would depend on `snaps-core`.

### Option B: Harmonic Engine as Central API
Make Harmonic Engine the single source of truth, expose via:
- MCP for AI assistants (already done)
- HTTP API for Python scripts
- NPM package for JS/TS projects

Python projects would call Harmonic Engine's HTTP API for all pitch class / scale / chord operations.

### Option C: JSON as Interchange (Current Approach)
Keep code separate but ensure JSON data files are identical:
- Single source of truth for `pressing_scale_dict.json`
- Single source of truth for `chords_no_supersets.json`
- Scripts copy these files when needed

**Current state:** Option C (ad-hoc), should move toward Option A or B.

---

## Workflow Integration Ideas

### 1. Composition Workflow
```
1. Define song structure in Scale Navigator Sketchpad
   → Creates JSON with nodes (harmonic states) and edges (transitions)

2. Run sketchpad_matcher to find sample candidates per transition
   → Creates alpine-lady_transitions.json

3. Human selection: choose which samples to use for each transition
   → Populate selected_samples field

4. Run arranger in "constrained mode" to render
   → Uses pre-selected samples but adds layers (bass flute, organetta)
```

### 2. Exploration Workflow
```
1. Run arranger in generative mode
   → Produces chain with inferred harmony

2. Export chord progression to Sketchpad format
   → Creates explorable progression in Scale Navigator

3. Refine in Sketchpad, re-render with constraints
```

### 3. Hybrid Workflow
```
1. Define harmonic waypoints (not full progression)
   → "Start on CM7, pass through F#7, end on DM7"

2. Arranger builds chain that honors waypoints
   → Free exploration between waypoints, constrained at checkpoints
```

---

## Arranger-Specific Notes (This Session)

### Scelsi Pezzi Layer
- Added as `LayerType.INTERVAL` (like Organetta)
- Transposes single-pitch sustaining phrases to fit harmony
- Default interval: 8 seconds
- Selection: `SelectionMode.SEQUENTIAL`

### Progressions as Skeleton Samples
- glaz_sax, hyacinthe, kraus are **skeleton samples**
- They participate in chain building, concatenated sequentially
- They are NOT overlay layers
- Their `chord_sequence` events populate the harmonic timeline

### Manual Chord Timings
- Stored in `samples_data.json` per library
- `chord_sequence` field with `start_time` for each chord
- Used during chain render to know when harmony changes

---

## Action Items

1. **Consolidate library paths** - Create single registry used by all projects
2. **Share chord dictionary** - Ensure `chords_no_supersets.json` is identical everywhere
3. **Document manifest format** - Formalize `samples_data.json` schema
4. **Add constrained mode to Arranger** - Honor pre-selected samples from sketchpad_matcher
5. **Export progression from Arranger** - Generate Sketchpad-compatible JSON from chain output

---

## File Locations Quick Reference

| Resource | Path |
|----------|------|
| Librarian | `/Users/soney/Github/librarian/` |
| Harmonic Engine | `/Users/soney/Github/harmonic-engine/` |
| Arranger | `/Users/soney/Github/arranger/` |
| Sample libraries | `/Users/soney/Music/samples/` |
| Chord dictionary | `/Users/soney/Github/arranger/data/chords_no_supersets.json` |
| Scale dictionary | `/Users/soney/Github/librarian/data/pressing_scale_dict.json` |
| Songs (Sketchpad outputs) | `/Users/soney/Github/songs/` |
