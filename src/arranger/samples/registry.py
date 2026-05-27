"""
Sample library path definitions.

All sample directories, audio subdirs, and manifest paths are defined here.
"""

from pathlib import Path

# =============================================================================
# PRIMARY SAMPLE LIBRARIES (for skeleton chain)
# =============================================================================

# Feldman 3voices (primary)
SAMPLES_DIR = Path("/Users/soney/Music/samples/3voices-feldman")
AUDIO_DIR = SAMPLES_DIR / "samples"
MANIFEST_PATH = SAMPLES_DIR / "samples_data.json"

# Strings Handel (secondary - can be mixed in)
HANDEL_DIR = Path("/Users/soney/Music/samples/strings_handel")
HANDEL_AUDIO_DIR = HANDEL_DIR / "samples"
HANDEL_MANIFEST_PATH = HANDEL_DIR / "samples_data.json"

# =============================================================================
# CONTINUOUS LAYERS (one sample at a time, back-to-back)
# =============================================================================

# Bass flute one-shots
BASSFLUTE_DIR = Path("/Users/soney/Music/samples/bassflute")
BASSFLUTE_AUDIO_DIR = BASSFLUTE_DIR / "samples"
BASSFLUTE_MANIFEST_PATH = BASSFLUTE_DIR / "samples_data.json"

# Trichords (continuous, like bass flute)
TRICHORDS_DIR = Path("/Users/soney/Music/samples/trichords")
TRICHORDS_AUDIO_DIR = TRICHORDS_DIR / "samples"
TRICHORDS_MANIFEST_PATH = TRICHORDS_DIR / "samples_data.json"


# Avery Violin Phrase (continuous, like bass flute)
AVERYVIOLIN_DIR = Path("/Users/soney/Music/samples/violin_avery_phrase")
AVERYVIOLIN_AUDIO_DIR = AVERYVIOLIN_DIR / "samples"
AVERYVIOLIN_MANIFEST_PATH = AVERYVIOLIN_DIR / "samples_data.json"

# Dictamel (continuous, like bass flute)
DICTAMEL_DIR = Path("/Users/soney/Music/samples/dictamel")
DICTAMEL_AUDIO_DIR = DICTAMEL_DIR / "samples"
DICTAMEL_MANIFEST_PATH = DICTAMEL_DIR / "samples_data.json"

# Scelsi Pezzi (continuous, sustaining single-pitch phrases)
SCELSIPEZZI_DIR = Path("/Users/soney/Music/samples/scelsi_pezzi")
SCELSIPEZZI_AUDIO_DIR = SCELSIPEZZI_DIR / "samples"
SCELSIPEZZI_MANIFEST_PATH = SCELSIPEZZI_DIR / "samples_data.json"

# =============================================================================
# INTERVAL LAYERS (fire every N seconds)
# =============================================================================

# Organetta one-shots (every 4 seconds, alphanumeric order)
ORGANETTA_DIR = Path("/Users/soney/Music/samples/organetta")
ORGANETTA_AUDIO_DIR = ORGANETTA_DIR / "samples"
ORGANETTA_MANIFEST_PATH = ORGANETTA_DIR / "samples_data.json"

# Feedback loops (overlapping, random selection)
FEEDBACK_DIR = Path("/Users/soney/Music/samples/feedback")
FEEDBACK_AUDIO_DIR = FEEDBACK_DIR / "samples"
FEEDBACK_MANIFEST_PATH = FEEDBACK_DIR / "samples_data.json"

# Stylo (every 2 seconds, like organetta but faster)
STYLO_DIR = Path("/Users/soney/Music/samples/stylo")
STYLO_AUDIO_DIR = STYLO_DIR / "samples"
STYLO_MANIFEST_PATH = STYLO_DIR / "samples_data.json"

# Tremolo Oct (interval layer, like organetta)
TREMOLO_OCT_DIR = Path("/Users/soney/Music/samples/tremolo_oct")
TREMOLO_OCT_AUDIO_DIR = TREMOLO_OCT_DIR / "samples"
TREMOLO_OCT_MANIFEST_PATH = TREMOLO_OCT_DIR / "samples_data.json"

# =============================================================================
# CLOUD LAYERS (rapid fire bursts)
# =============================================================================

# Brodero one-shots (rapid fire)
BRODERO_DIR = Path("/Users/soney/Music/samples/brodero")
BRODERO_MANIFEST_PATH = BRODERO_DIR / "samples_data.json"

# Jicello expanded (distributed slowly)
JICELLO_DIR = Path("/Users/soney/Music/samples/jicelloexpanded")
JICELLO_AUDIO_DIR = JICELLO_DIR / "samples"
JICELLO_MANIFEST_PATH = JICELLO_DIR / "samples_data.json"

# Gothic Harp one-shots (16th note clouds)
GOTHICHARP_DIR = Path("/Users/soney/Music/samples/gothic_harp")
GOTHICHARP_AUDIO_DIR = GOTHICHARP_DIR / "samples"
GOTHICHARP_MANIFEST_PATH = GOTHICHARP_DIR / "samples_data.json"

# Gentle Harpsichord one-shots (32nd note clouds)
GENTLEHARPSI_DIR = Path("/Users/soney/Music/samples/GentleHarpsichord")
GENTLEHARPSI_AUDIO_DIR = GENTLEHARPSI_DIR / "samples"
GENTLEHARPSI_MANIFEST_PATH = GENTLEHARPSI_DIR / "samples_data.json"

# Laken samples (single-note, similar to gothic harp)
LAKEN_DIR = Path("/Users/soney/Music/samples/laken")
LAKEN_AUDIO_DIR = LAKEN_DIR / "samples"
LAKEN_MANIFEST_PATH = LAKEN_DIR / "samples_data.json"

# =============================================================================
# RHYTHMIC LAYERS
# =============================================================================

# MinorChordBeat one-shots (eighth notes at 120 BPM = 250ms)
MINORCHORDBEAT_DIR = Path("/Users/soney/Music/samples/minorchordbeat")
MINORCHORDBEAT_AUDIO_DIR = MINORCHORDBEAT_DIR / "samples"
MINORCHORDBEAT_MANIFEST_PATH = MINORCHORDBEAT_DIR / "samples_data.json"

# MuteBowl one-shots (eighth notes at 120 BPM = 250ms)
MUTEBOWL_DIR = Path("/Users/soney/Music/samples/mutebowl")
MUTEBOWL_AUDIO_DIR = MUTEBOWL_DIR / "samples"
MUTEBOWL_MANIFEST_PATH = MUTEBOWL_DIR / "samples_data.json"

# =============================================================================
# CHORD/QUALITY LAYERS
# =============================================================================

# Major/Minor chord samples (quality-aware layer)
MINORCHORDS_DIR = Path("/Users/soney/Music/samples/minor-chords")
MINORCHORDS_AUDIO_DIR = MINORCHORDS_DIR / "samples"
MINORCHORDS_MANIFEST_PATH = MINORCHORDS_DIR / "samples_data.json"

MAJORCHORDS_DIR = Path("/Users/soney/Music/samples/major-chords")
MAJORCHORDS_AUDIO_DIR = MAJORCHORDS_DIR / "samples"
MAJORCHORDS_MANIFEST_PATH = MAJORCHORDS_DIR / "samples_data.json"

# =============================================================================
# SINGLE-NOTE LAYERS
# =============================================================================

# Prophet False one-shots (single notes)
PROPHETFALSE_DIR = Path("/Users/soney/Music/samples/prophet_false")
PROPHETFALSE_AUDIO_DIR = PROPHETFALSE_DIR / "samples"
PROPHETFALSE_MANIFEST_PATH = PROPHETFALSE_DIR / "samples_data.json"

# Harmonicker one-shots (harmonica chords/intervals)
HARMONICKER_DIR = Path("/Users/soney/Music/samples/Harmonicker")
HARMONICKER_AUDIO_DIR = HARMONICKER_DIR / "samples"
HARMONICKER_MANIFEST_PATH = HARMONICKER_DIR / "samples_data.json"

# =============================================================================
# CHORD-TRIGGERED LAYERS (play on chord changes if fitting sample exists)
# =============================================================================

# Godette samples (one at a time, triggers on chord changes)
GODETTE_DIR = Path("/Users/soney/Music/samples/godette-samples")
GODETTE_AUDIO_DIR = GODETTE_DIR / "samples"
GODETTE_MANIFEST_PATH = GODETTE_DIR / "samples_data.json"

# =============================================================================
# PROGRESSION SAMPLES (multi-chord sequences)
# =============================================================================

GLAZ_SAX_DIR = Path("/Users/soney/Music/samples/glaz_sax_chorales")
GLAZ_SAX_AUDIO_DIR = GLAZ_SAX_DIR / "samples"
GLAZ_SAX_MANIFEST_PATH = GLAZ_SAX_DIR / "samples_data.json"

HYACINTHE_DIR = Path("/Users/soney/Music/samples/hyacinthe")
HYACINTHE_AUDIO_DIR = HYACINTHE_DIR / "samples"
HYACINTHE_MANIFEST_PATH = HYACINTHE_DIR / "samples_data.json"

KRAUS_DIR = Path("/Users/soney/Music/samples/KrausChorale")
KRAUS_AUDIO_DIR = KRAUS_DIR / "samples"
KRAUS_MANIFEST_PATH = KRAUS_DIR / "samples_data.json"

# =============================================================================
# TRANSPOSITION CONSTANTS
# =============================================================================

MIN_TRANSPOSITION = -9
MAX_TRANSPOSITION = 5

# =============================================================================
# TIMING CONSTANTS
# =============================================================================

GLISSANDO_MS = 200  # 200ms portamento glide duration
GLISSANDO_ANTICIPATION_MS = 100  # Start glide 100ms earlier than note onset

# =============================================================================
# AUDIO CONSTANTS
# =============================================================================

SAMPLE_NORMALIZE_DB = -5.0  # Peak normalize to -5 dB

# =============================================================================
# DATA PATHS
# =============================================================================

# Output directory (relative to repo root)
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "output"

# Chord dictionary (local copy - rich chord vocabulary with 30k+ chords)
CHORDS_JSON_PATH = Path(__file__).parent.parent.parent.parent / "data" / "chords_no_supersets.json"

# =============================================================================
# MUSIC THEORY CONSTANTS
# =============================================================================

NOTE_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
