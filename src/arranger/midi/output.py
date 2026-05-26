"""Output functions for rendering graphs and exporting scores."""

import os
from pathlib import Path

import graphviz


def render_graph(arr_dict, output_path=None):
    """Create a Graphviz directed graph and optionally save/open it.

    Args:
        arr_dict: Analysis dictionary with children relationships
        output_path: Optional path for output file (without extension)

    Returns:
        The graphviz.Digraph object
    """
    graph = graphviz.Digraph(engine="fdp", graph_attr={"size": "8.5, 11"})
    for name, node in arr_dict.items():
        graph.node(name, name)
        for child in node["children"]:
            graph.edge(name, child)

    if output_path:
        graph.render(output_path, view=False)
    else:
        graph.render(view=True)

    return graph


def export_lilypond(part_with_measures, output_path=None):
    """Export a music21 Part to LilyPond format.

    Args:
        part_with_measures: A music21 Part or Score object
        output_path: Path for output .ly file (default: outputs/scores/output.ly)

    Returns:
        Path to the generated LilyPond file
    """
    if output_path is None:
        output_dir = Path("outputs/scores")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "output.ly"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    part_with_measures.write('lilypond', fp=str(output_path))
    print(f"LilyPond file written to: {output_path}")
    return output_path


def export_musicxml(part_with_measures, output_path=None):
    """Export a music21 Part to MusicXML format.

    Args:
        part_with_measures: A music21 Part or Score object
        output_path: Path for output .musicxml file (default: outputs/scores/output.musicxml)

    Returns:
        Path to the generated MusicXML file
    """
    if output_path is None:
        output_dir = Path("outputs/scores")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "output.musicxml"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    part_with_measures.write('musicxml', fp=str(output_path))
    print(f"MusicXML file written to: {output_path}")
    return output_path


def show_musicxml(part_with_measures):
    """Open the score in the default MusicXML viewer (e.g., MuseScore).

    Args:
        part_with_measures: A music21 Part or Score object
    """
    part_with_measures.show("musicxml")
