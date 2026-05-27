#!/usr/bin/env python3
"""CLI for MIDI progression concatenation."""

import argparse
import random
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arranger.midi import (
    get_filelist,
    load_or_build_analysis,
    build_connections,
    prune_sinks,
    build_non_sink_children,
    random_walk,
    render_graph,
    export_lilypond,
    export_musicxml,
    show_musicxml,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Algorithmically append a series of modulatory chord progressions into a loop."
    )
    parser.add_argument(
        "start",
        help="starting MIDI filename (relative path under datasets directory)",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=100,
        help="target number of progressions to chain (approximate, not exact)",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="render a Graphviz graph of progression connections",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="disable realtime MIDI playback while generating progressions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for reproducible walks",
    )
    parser.add_argument(
        "--output-format",
        choices=["lilypond", "musicxml", "show"],
        default="lilypond",
        help="output format: lilypond (default), musicxml, or show (opens in viewer)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output file path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path(__file__).parent.parent / "datasets",
        help="directory containing MIDI datasets",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    datasets_dir = args.datasets_dir.resolve()
    if not datasets_dir.exists():
        raise SystemExit(f"Datasets directory not found: {datasets_dir}")

    filelist = get_filelist(datasets_dir)
    if not filelist:
        raise SystemExit(f"No .mid or .midi files found under {datasets_dir}")

    # Resolve start path
    start_path = args.start
    if not Path(start_path).is_absolute():
        # Try to find it in datasets
        candidates = [f for f in filelist if start_path in f]
        if len(candidates) == 1:
            start_path = candidates[0]
        elif len(candidates) > 1:
            # Prefer exact match
            exact = [f for f in candidates if f.endswith(start_path)]
            if exact:
                start_path = exact[0]
            else:
                print(f"Multiple matches for '{start_path}':")
                for c in candidates[:10]:
                    print(f"  {c}")
                raise SystemExit("Please specify a more precise path.")
        else:
            raise SystemExit(f"Start file '{start_path}' not found in datasets.")

    arr_dict = load_or_build_analysis(filelist)

    if start_path not in arr_dict:
        raise SystemExit(
            f"Start file '{start_path}' is not usable "
            "(it may have no chords or failed to parse)."
        )

    build_connections(arr_dict)

    if args.graph:
        render_graph(arr_dict)

    prune_sinks(arr_dict)
    build_non_sink_children(arr_dict)

    part_with_measures = random_walk(
        arr_dict,
        start_name=start_path,
        target_progressions=args.num,
        enable_playback=not args.no_play,
    )

    # Handle output
    if args.output_format == "lilypond":
        export_lilypond(part_with_measures, args.output)
    elif args.output_format == "musicxml":
        export_musicxml(part_with_measures, args.output)
    else:  # show
        show_musicxml(part_with_measures)


if __name__ == "__main__":
    main()
