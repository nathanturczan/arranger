"""Graph construction and traversal for MIDI progression networks."""


def build_connections(arr_dict):
    """Populate children / parents relationships based on interval matching.

    For 4–6 voice chords we allow flexible matching: any contiguous 4-note
    window (bottom/middle/top) can be the SATB frame, and we match on the
    (tenor–alto, alto–soprano) interval pairs for those windows.

    If either side doesn't have a 4-note window (e.g. 3-voice boundary),
    we fall back to comparing upper intervals derived from the full chord.
    """
    for name_1, value_1 in arr_dict.items():
        for name_2, value_2 in arr_dict.items():
            begin_chord = value_2["first_chord"]
            end_chord = value_1["last_chord"]

            begin_patterns = begin_chord.get("match_patterns", [])
            end_patterns = end_chord.get("match_patterns", [])

            matched = False

            if begin_patterns and end_patterns:
                # 4–6 voice flexible matching: any overlapping 4-voice slice can match
                for ep in end_patterns:
                    if ep in begin_patterns:
                        matched = True
                        break
            else:
                # fallback: compare upper adjacent intervals (old behavior)
                if end_chord["intervals"][1:] == begin_chord["intervals"][1:]:
                    matched = True

            if matched:
                value_1["children"].append(name_2)
                value_2["parents"].append(name_1)


def mark_sinks(arr_dict):
    """Mark nodes as sinks if all their children are already sinks (or self)."""
    done = True
    for name, node in arr_dict.items():
        if all(arr_dict[child].get("sink") for child in node["children"] if child != name):
            if not node.get("sink"):
                done = False
            node["sink"] = True
    return done


def prune_sinks(arr_dict):
    """Loop mark_sinks over and over again until no new sinks appear."""
    done = False
    while not done:
        done = mark_sinks(arr_dict)


def build_non_sink_children(arr_dict):
    """Assemble list of children that are not sinks for each node."""
    for name, node in arr_dict.items():
        non_sink_children = []
        node["non_sink_children"] = non_sink_children
        for child in node["children"]:
            if child != name and not arr_dict[child].get("sink"):
                non_sink_children.append(child)
