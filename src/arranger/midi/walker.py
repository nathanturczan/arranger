"""Random walk through progression graphs to generate new compositions."""

import os
import random

from music21 import chord, expressions, instrument, midi, note, stream, tie

from .analyzer import get_parsed


def add_ties_for_repeated_notes(part_stream):
    """
    Scan the Part/Score and, for each MIDI pitch, tie together
    immediately consecutive occurrences (no time gap between them).

    Works for both Chord and Note objects.
    """
    s = part_stream
    flat = s.flatten()  # global timeline

    prev_note_by_midi = {}
    prev_end_by_midi = {}

    for el in flat.notes:  # Note or Chord, in time order
        offset = el.offset
        end = offset + el.quarterLength

        if isinstance(el, chord.Chord):
            notes_iter = el.notes
        else:
            notes_iter = [el]

        for n in notes_iter:
            midi_val = n.pitch.midi
            prev_end = prev_end_by_midi.get(midi_val)

            # If the previous note of this MIDI pitch ends exactly where this starts,
            # add a tie between them.
            if prev_end is not None and abs(prev_end - offset) < 1e-6:
                prev_note = prev_note_by_midi[midi_val]

                if prev_note.tie is None:
                    prev_note.tie = tie.Tie("start")
                elif prev_note.tie.type in ("stop", "end"):
                    prev_note.tie = tie.Tie("continue")
                else:
                    prev_note.tie.type = "continue"

                if n.tie is None:
                    n.tie = tie.Tie("stop")
                else:
                    n.tie.type = "stop"

            prev_note_by_midi[midi_val] = n
            prev_end_by_midi[midi_val] = end


def random_walk(arr_dict, start_name, target_progressions=100, enable_playback=True,
                output_format="lilypond", output_path=None):
    """
    Walk through the graph of progressions and build a score.

    We treat target_progressions as a *hint*:
    - Try to get back to start_name after roughly that many steps.
    - The final progression in the output is start_name again (so it's loopable).

    Args:
        arr_dict: Analysis dictionary from load_or_build_analysis
        start_name: Starting MIDI filename
        target_progressions: Target number of progressions to chain (approximate)
        enable_playback: Enable realtime MIDI playback while generating
        output_format: "lilypond" (default), "musicxml", or "show"
        output_path: Path for output file (auto-generated if None)

    Returns:
        The generated Part with measures
    """
    if start_name not in arr_dict:
        raise ValueError(f"Start file '{start_name}' not found in analysis.")

    original_start = start_name
    current_name = start_name

    # soft bounds for "how long" the loop should be
    min_steps = max(1, int(target_progressions * 0.75))
    max_steps = max(min_steps + 1, int(target_progressions * 1.5))

    transposer = 0
    paper = []
    part = stream.Part()
    # set instrument for MuseScore (change to whatever you want)
    part.insert(0, instrument.Viola())
    num_steps = 0

    for _ in range(max_steps):
        name = current_name
        num_steps += 1

        print(name)

        prog_stream = get_parsed(name)

        # Try to take the first notated measure (1-based in music21).
        # If that gives an empty slice, fall back to the whole stream.
        c = prog_stream.measures(1, 1)
        if len(c.recurse().notesAndRests) == 0:
            c = prog_stream

        # normalization:
        # - on the very first snippet, transpose so that the outer voices
        #   (bass and soprano) are centered around middle C (MIDI 60).
        # - for later snippets, keep the older "bass → 48" heuristic.
        bass = arr_dict[name]["first_chord"]["bass"]
        soprano = arr_dict[name]["first_chord"]["soprano"]

        if num_steps == 1:
            midpoint = (bass + soprano) / 2.0
            desired_midpoint = 60
            normalizer = int(round(desired_midpoint - midpoint))
        else:
            normalizer = 48 - bass

        end_soprano = arr_dict[name]["last_chord"]["soprano"] + transposer
        begin_soprano = arr_dict[name]["first_chord"]["soprano"] + transposer

        temp_list = (
            name,
            normalizer,
            transposer,
            begin_soprano + normalizer,
            end_soprano + normalizer,
        )
        paper.append(temp_list)
        li = range(len(paper))

        # transpose + normalize
        c = c.transpose(normalizer)
        c = c.transpose(transposer)

        transposer += arr_dict[name]["transposer"]
        for i, item in enumerate(li):
            if i > 0:
                octave_check = paper[li[i - 1]][4] - paper[item][3]
                c = c.transpose(octave_check)

        # insert a text label with the snippet's name at its start
        label = os.path.splitext(os.path.basename(name))[0]
        current_offset = part.duration.quarterLength
        txt = expressions.TextExpression(label)
        part.insert(current_offset, txt)

        # chordified stream – append chords/notes/rests directly to the Part
        printout = c.chordify()
        for el in printout.flat:
            if isinstance(el, (chord.Chord, note.Note, note.Rest)):
                part.append(el)

        if enable_playback:
            sp = midi.realtime.StreamPlayer(c)
            sp.play()

        # if we've come back to the starting progression after at least 2 steps
        # and we're past the minimum suggested length, stop here
        if num_steps > 1 and num_steps >= min_steps and name == original_start:
            break

        # otherwise, pick the next progression
        children = arr_dict[name]["non_sink_children"]
        if not children:
            # total dead end: pick anything
            candidates = list(arr_dict.keys())
        else:
            candidates = list(children)

        # heuristic: once we've hit min_steps, if we *can* go to original_start next,
        # bias toward that so we can close the loop.
        if num_steps >= min_steps and original_start in candidates:
            current_name = original_start
        else:
            current_name = random.choice(candidates)

    # tie any immediately repeated notes (same MIDI pitch, back-to-back)
    # do this on the raw Part, where all notes live directly in that stream
    add_ties_for_repeated_notes(part)

    # now create measures so notation + MusicXML are valid
    part_with_measures = part.makeMeasures()

    return part_with_measures
