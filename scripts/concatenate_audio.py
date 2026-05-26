#!/usr/bin/env python3
"""CLI for audio sample concatenation via scale network navigation."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from concatenator.audio import (
    concatenate_audio,
    organize_samples,
    generate_companion_midi,
    generate_score_report,
)
from concatenator.audio.projects import (
    list_projects,
    load_project,
    get_scales_dir,
    get_output_dir,
    format_sample_name,
)
from concatenator.scales import load_scales_data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Concatenate audio samples via Tymoczko 57-scale network navigation."
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # list-projects command
    list_parser = subparsers.add_parser(
        "list-projects",
        help="List available audio concatenation projects"
    )

    # run command (project-based)
    run_parser = subparsers.add_parser(
        "run",
        help="Run a configured project"
    )
    run_parser.add_argument(
        "project",
        help="Project name (e.g., 'sarabande', 'wildrose')",
    )
    run_parser.add_argument(
        "-n", "--num",
        type=int,
        help="Number of measures (default: from project config)",
    )
    run_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output WAV path (default: project output dir)",
    )
    run_parser.add_argument(
        "--midi",
        action="store_true",
        help="Also generate companion MIDI file (default: on)",
    )
    run_parser.add_argument(
        "--no-midi",
        action="store_true",
        help="Skip MIDI generation",
    )
    run_parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip score report generation",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible output",
    )
    run_parser.add_argument(
        "--json",
        action="store_true",
        help="Output journey data as JSON",
    )

    # organize command
    org_parser = subparsers.add_parser(
        "organize",
        help="Organize samples into scale folders (run once before concatenating)"
    )
    org_parser.add_argument(
        "samples_dir",
        type=Path,
        help="Directory containing source WAV samples",
    )
    org_parser.add_argument(
        "manifest",
        type=Path,
        help="Path to samples_data.json with pitch class info",
    )
    org_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory for scales_dir (default: samples_dir/scales_dir)",
    )
    org_parser.add_argument(
        "--no-symlinks",
        action="store_true",
        help="Copy files instead of creating symlinks",
    )
    org_parser.add_argument(
        "--transpositions",
        type=str,
        default="-7,-6,-5,-4,-3,-2,-1,1,2,3,4",
        help="Comma-separated semitone transpositions (default: -7 to +4 excluding 0)",
    )

    # chain command (low-level)
    chain_parser = subparsers.add_parser(
        "chain",
        help="Chain samples into a continuous audio file (low-level)"
    )
    chain_parser.add_argument(
        "scales_dir",
        type=Path,
        help="Path to organized scales_dir",
    )
    chain_parser.add_argument(
        "prefix",
        help="Sample naming prefix (e.g., 'wildrose', 'takemitsu_perc')",
    )
    chain_parser.add_argument(
        "-n", "--num",
        type=int,
        default=50,
        help="Number of measures/segments to chain (default: 50)",
    )
    chain_parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output WAV path (default: auto-generated)",
    )
    chain_parser.add_argument(
        "--midi",
        action="store_true",
        help="Also generate companion MIDI file showing scale pitches",
    )
    chain_parser.add_argument(
        "--crossfade",
        type=int,
        default=1,
        help="Crossfade duration in milliseconds (default: 1)",
    )
    chain_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible output",
    )
    chain_parser.add_argument(
        "--json",
        action="store_true",
        help="Output journey data as JSON",
    )

    return parser.parse_args(argv)


def cmd_list_projects(args):
    """List available projects."""
    projects = list_projects()
    if not projects:
        print("No projects found.")
        return

    print("Available projects:")
    for name in projects:
        try:
            config = load_project(name)
            desc = config.get("description", "")
            path_exists = "✓" if config["path"].exists() else "✗"
            print(f"  {name:15} {path_exists} {desc}")
        except Exception as e:
            print(f"  {name:15} ✗ (error: {e})")


def cmd_run(args):
    """Run a configured project."""
    config = load_project(args.project)

    scales_dir = get_scales_dir(config)
    if not scales_dir.exists():
        raise SystemExit(
            f"scales_dir not found at {scales_dir}\n"
            f"Run sample_clerk.py in {config['path']} first."
        )

    num_samples = args.num or config["num_samples"]
    crossfade = config.get("crossfade_ms", 1)

    # MIDI and report are on by default
    generate_midi = not args.no_midi
    generate_report = not args.no_report

    if args.seed is not None:
        random.seed(args.seed)

    # Use the project's existing make_song.py logic via our chain function
    output_path, journey = concatenate_audio(
        scales_dir=scales_dir,
        sample_prefix=config["sample_prefix"],
        num_measures=num_samples,
        output_path=args.output,
        crossfade_ms=crossfade,
        seed=args.seed,
        verbose=not args.json,
    )

    midi_path = None
    report_path = None

    # Generate MIDI (precisely timed to audio chunks)
    if generate_midi:
        scales_data = load_scales_data()
        midi_path = output_path.with_suffix(".mid")
        generate_companion_midi(journey, scales_data, midi_path)
        if not args.json:
            print(f"MIDI exported: {midi_path}")

    # Generate score report (markdown with pitch class info)
    if generate_report:
        report_path = output_path.with_suffix(".md")
        generate_score_report(
            journey,
            report_path,
            title=f"{args.project.title()} - Audio Concatenation Score"
        )
        if not args.json:
            print(f"Score report: {report_path}")

    if args.json:
        result = {
            "project": args.project,
            "output_path": str(output_path),
            "journey": journey,
        }
        if midi_path:
            result["midi_path"] = str(midi_path)
        if report_path:
            result["report_path"] = str(report_path)
        print(json.dumps(result, indent=2))


def cmd_organize(args):
    """Handle the organize command."""
    transpositions = tuple(int(x) for x in args.transpositions.split(","))

    output_dir = organize_samples(
        samples_dir=args.samples_dir,
        samples_manifest=args.manifest,
        output_dir=args.output,
        transpositions=transpositions,
        use_symlinks=not args.no_symlinks,
        verbose=True,
    )

    print(f"\nScales directory created at: {output_dir}")
    print(f"Ready to concatenate with: concatenate_audio.py chain {output_dir} <prefix>")


def cmd_chain(args):
    """Handle the chain command."""
    output_path, journey = concatenate_audio(
        scales_dir=args.scales_dir,
        sample_prefix=args.prefix,
        num_measures=args.num,
        output_path=args.output,
        crossfade_ms=args.crossfade,
        seed=args.seed,
        verbose=not args.json,
    )

    if args.midi:
        scales_data = load_scales_data()
        midi_path = output_path.with_suffix(".mid")
        generate_companion_midi(journey, scales_data, midi_path)
        if not args.json:
            print(f"MIDI exported: {midi_path}")

    if args.json:
        result = {
            "output_path": str(output_path),
            "journey": journey,
        }
        if args.midi:
            result["midi_path"] = str(midi_path)
        print(json.dumps(result, indent=2))


def main(argv=None):
    args = parse_args(argv)

    if args.command == "list-projects":
        cmd_list_projects(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "organize":
        cmd_organize(args)
    elif args.command == "chain":
        cmd_chain(args)
    else:
        print("Usage: concatenate_audio.py {list-projects,run,organize,chain} ...")
        print("\nCommands:")
        print("  list-projects  List available audio projects")
        print("  run            Run a configured project")
        print("  organize       Organize samples into scale folders")
        print("  chain          Low-level sample chaining")
        print("\nRun with -h for help")
        sys.exit(1)


if __name__ == "__main__":
    main()
