"""MIDI analysis functions for chord extraction and interval calculation."""

import os
import pickle
from pathlib import Path

import numpy as np
from music21 import converter


# Cache parsed MIDI streams so we don't keep re-reading from disk
_parsed_cache = {}


def get_parsed(name):
    """Return a cached music21 stream for the given MIDI filename."""
    if name not in _parsed_cache:
        _parsed_cache[name] = converter.parse(name)
    return _parsed_cache[name]


def build_intervals(chord):
    """Extract SATB-ish info from a chord, supporting 4–6 voices.

    We keep:
    - midi: sorted list of distinct MIDI pitches
    - bass / tenor / alto / soprano: outer and inner voices (best effort if < 4 notes)
    - intervals: full adjacent-interval list over all notes
    - match_patterns: list of (T–A, A–S) interval pairs over all possible
      4-note windows, used to match 4–6 voice chords flexibly (top/middle/bottom).
    """
    # chord.pitches elements are already Pitch objects, so just use .midi
    midi = sorted({p.midi for p in chord.pitches})

    if not midi:
        return {
            "midi": [],
            "bass": None,
            "tenor": None,
            "alto": None,
            "soprano": None,
            "intervals": (),
            "match_patterns": [],
        }

    bass = midi[0]
    # if we have fewer than 3 notes, fall back sensibly
    if len(midi) > 2:
        tenor = midi[1]
        alto = midi[-2]
    else:
        tenor = bass
        alto = midi[-1]
    soprano = midi[-1]

    intervals = tuple(int(x) for x in np.diff(midi)) if len(midi) > 1 else ()

    # build 4-note window patterns for flexible matching of 5–6 voice chords
    match_patterns = []
    if len(midi) >= 4:
        for i in range(len(midi) - 3):
            window = midi[i : i + 4]
            diffs = np.diff(window)
            # pattern for this window: the upper two adjacent intervals (T–A, A–S)
            if len(diffs) >= 3:
                match_patterns.append((int(diffs[1]), int(diffs[2])))

    return {
        "midi": midi,
        "bass": bass,
        "tenor": tenor,
        "alto": alto,
        "soprano": soprano,
        "intervals": intervals,
        "match_patterns": match_patterns,
    }


def get_filelist(root: Path = Path(".")):
    """Return list of .mid / .midi filenames recursively under the given root."""
    midi_files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".mid", ".midi"):
            midi_files.append(str(p))
    return sorted(midi_files)


def load_or_build_analysis(filelist, analysis_path="analysis.pkl"):
    """Load cached analysis if possible; otherwise compute and cache it.

    We store both the analysis dict and a simple metadata dict of
    {filename: mtime} so we can detect when files have changed.
    """
    stored_meta = None
    arr_dict = {}

    if os.path.exists(analysis_path):
        try:
            with open(analysis_path, "rb") as fp:
                loaded = pickle.load(fp)
            # Backwards compatibility: older pickles may only contain arr_dict
            if isinstance(loaded, tuple) and len(loaded) == 2:
                stored_meta, arr_dict = loaded
            else:
                arr_dict = loaded
        except Exception as e:
            print(f"Warning: could not load cached analysis ({e}); rebuilding.")
            stored_meta = None
            arr_dict = {}

    current_meta = {f: os.path.getmtime(f) for f in filelist}

    # If we have analysis and the meta matches, reuse it
    if arr_dict and stored_meta == current_meta:
        return arr_dict

    # Otherwise, rebuild analysis from scratch
    arr_dict = {}
    for var in filelist:
        try:
            # convert to music21 stream
            mid = get_parsed(var)
            chords = mid.chordify()
            # RECURSE to find chords anywhere in the structure
            chord_elems = chords.recurse().getElementsByClass("Chord")

            if len(chord_elems) == 0:
                print(f"Skipping {var}: no chords found after chordify()")
                continue

            # get intervals of first and last chords of each progression
            first_chord = build_intervals(chord_elems[0])
            last_chord = build_intervals(chord_elems[-1])

            # skip snippets that begin AND end with a stable 3-voice texture
            if len(first_chord["midi"]) == 3 and len(last_chord["midi"]) == 3:
                print(f"Skipping {var}: 3-voice texture at both start and end.")
                continue

            current = {"children": [], "parents": []}
            current["first_chord"] = first_chord
            current["last_chord"] = last_chord

            # how much we need to transpose the NEXT progression (cumulative later)
            current["transposer"] = (
                current["last_chord"]["bass"] - current["first_chord"]["bass"]
            )
            arr_dict[var] = current

        except Exception as e:
            # keep going if one file is weird
            print(f"Skipping {var}: error during analysis ({e})")
            continue

    if not arr_dict:
        raise SystemExit("No usable MIDI files found (all failed analysis).")

    with open(analysis_path, "wb") as fp:
        pickle.dump((current_meta, arr_dict), fp)

    return arr_dict
