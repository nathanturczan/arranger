#!/usr/bin/env python3
"""
Build Idiom Index for Harmonic Engine

Parses the MIDI corpus (905 files from textbooks) and extracts:
- Source metadata (from filename)
- First/last chord pitch classes
- First/last chord intervals
- Voice-leading info

Outputs: idioms_index.json for querying via Harmonic Engine MCP
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict

from music21 import converter, chord as m21chord
import numpy as np


def parse_source_from_filename(filepath: str) -> dict:
    """Extract source metadata from filename.

    Examples:
        Schoenberg-p360-Ex297-Major-16.midi -> {source: "schoenberg", page: "360", example: "297", key: "major"}
        harmonielehre-p89-91-inversions-of-7th-chords.midi -> {source: "schoenberg", page: "89-91", topic: "inversions-of-7th-chords"}
        1.mid (in reger/) -> {source: "reger", number: "1"}
    """
    filename = os.path.basename(filepath)
    parent_dir = os.path.basename(os.path.dirname(filepath))

    # Normalize source from parent directory
    source = parent_dir.lower().replace('-dataset', '').replace('_', '-')
    if not source or source == 'datasets':
        source = 'unknown'

    metadata = {'source': source, 'filename': filename, 'filepath': filepath}

    # Try to extract page numbers
    page_match = re.search(r'p(\d+(?:-\d+)?)', filename, re.IGNORECASE)
    if page_match:
        metadata['page'] = page_match.group(1)

    # Try to extract example numbers
    ex_match = re.search(r'Ex(\d+)', filename, re.IGNORECASE)
    if ex_match:
        metadata['example'] = ex_match.group(1)

    # Try to extract key (Major/Minor)
    if 'major' in filename.lower():
        metadata['mode'] = 'major'
    elif 'minor' in filename.lower():
        metadata['mode'] = 'minor'

    # Try to extract topic from harmonielehre filenames
    topic_match = re.search(r'-([a-z-]+)\.midi?$', filename, re.IGNORECASE)
    if topic_match and topic_match.group(1) not in ['major', 'minor']:
        metadata['topic'] = topic_match.group(1)

    # For simple numbered files (reger, etc)
    num_match = re.match(r'^(\d+[a-z]?)\.midi?$', filename, re.IGNORECASE)
    if num_match:
        metadata['number'] = num_match.group(1)

    return metadata


def build_chord_info(m21_chord) -> dict:
    """Extract chord information from a music21 Chord object."""
    midi_notes = sorted([p.midi for p in m21_chord.pitches])
    pitch_classes = sorted(list(set(p.midi % 12 for p in m21_chord.pitches)))

    # Calculate intervals between adjacent notes
    intervals = tuple(int(x) for x in np.diff(midi_notes)) if len(midi_notes) > 1 else ()

    # SATB-ish extraction (bass, inner voices, soprano)
    bass = midi_notes[0] if midi_notes else None
    soprano = midi_notes[-1] if midi_notes else None

    return {
        'midi_notes': midi_notes,
        'pitch_classes': pitch_classes,
        'intervals': intervals,
        'bass': bass,
        'soprano': soprano,
        'bass_pc': bass % 12 if bass else None,
    }


def analyze_midi_file(filepath: str) -> dict | None:
    """Analyze a single MIDI file and extract idiom data."""
    try:
        stream = converter.parse(filepath)
        chords = stream.chordify()
        chord_elems = list(chords.recurse().getElementsByClass('Chord'))

        if len(chord_elems) < 1:
            return None

        first_chord = build_chord_info(chord_elems[0])
        last_chord = build_chord_info(chord_elems[-1])

        # Calculate transposer (bass movement across the progression)
        transposer = 0
        if first_chord['bass'] and last_chord['bass']:
            transposer = last_chord['bass'] - first_chord['bass']

        # Extract source metadata from filename
        metadata = parse_source_from_filename(filepath)

        return {
            'id': os.path.splitext(os.path.basename(filepath))[0],
            'source': metadata['source'],
            'metadata': metadata,
            'first_chord': first_chord,
            'last_chord': last_chord,
            'transposer': transposer,
            'num_chords': len(chord_elems),
        }

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def get_midi_files(root: Path) -> list[str]:
    """Recursively find all MIDI files under a root directory."""
    midi_files = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.mid', '.midi'):
            # Skip files in 'source' directories (LilyPond sources)
            if 'source' not in str(p):
                midi_files.append(str(p))
    return sorted(midi_files)


def main():
    # Find datasets directory
    script_dir = Path(__file__).parent
    datasets_dir = script_dir.parent / 'datasets'

    if not datasets_dir.exists():
        print(f"Datasets directory not found: {datasets_dir}")
        return

    print(f"Scanning {datasets_dir} for MIDI files...")
    midi_files = get_midi_files(datasets_dir)
    print(f"Found {len(midi_files)} MIDI files")

    # Analyze each file
    idioms = []
    errors = []

    for i, filepath in enumerate(midi_files):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(midi_files)}...")

        result = analyze_midi_file(filepath)
        if result:
            idioms.append(result)
        else:
            errors.append(filepath)

    print(f"\nSuccessfully analyzed {len(idioms)} idioms")
    print(f"Errors: {len(errors)}")

    # Group by source for summary
    by_source = defaultdict(list)
    for idiom in idioms:
        by_source[idiom['source']].append(idiom)

    print("\nBy source:")
    for source, source_idioms in sorted(by_source.items()):
        print(f"  {source}: {len(source_idioms)}")

    # Write output
    output_path = script_dir.parent / 'data' / 'idioms_index.json'
    output_path.parent.mkdir(exist_ok=True)

    output = {
        '_meta': {
            'description': 'Harmonic idiom index for querying via Harmonic Engine',
            'total_idioms': len(idioms),
            'sources': {source: len(idioms_list) for source, idioms_list in by_source.items()},
        },
        'idioms': idioms,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {output_path}")

    # Also write a copy to harmonic-engine/data for easy access
    he_data_path = Path('/Users/soney/Github/harmonic-engine/data/idioms_index.json')
    he_data_path.parent.mkdir(exist_ok=True)

    with open(he_data_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {he_data_path}")


if __name__ == '__main__':
    main()
